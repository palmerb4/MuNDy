// @HEADER
// **********************************************************************************************************************
//
//                                          Mundy: Multi-body Nonlocal Dynamics
//                                           Copyright 2024 Flatiron Institute
//                                                 Author: Bryce Palmer
//
// Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
//
// Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
// of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with Mundy. If not, see
// <https://www.gnu.org/licenses/>.
//
// **********************************************************************************************************************
// @HEADER

#ifndef MUNDY_ALENS_COMPUTE_MOBILITY_PERIPHERY_UTILS_EIGEN_MATRIX_PLUGIN_H_
#define MUNDY_ALENS_COMPUTE_MOBILITY_PERIPHERY_UTILS_EIGEN_MATRIX_PLUGIN_H_

inline void msgpack_unpack(msgpack::object o) {
    if(o.type != msgpack::type::ARRAY) { throw msgpack::type_error(); }

    msgpack::object * p = o.via.array.ptr;

    std::string type;
    *p >> type;
    if (type != "__eigen__") { throw msgpack::type_error(); }

    size_t rows;
    size_t cols;

    ++p;
    *p >> rows;
    ++p;
    *p >> cols;
    this->resize(rows, cols);

    for (int i = 0; i < this->cols(); ++i) {
        for (int j = 0; j < this->rows(); ++j) {
            ++p;
            *p >> this->operator()(j, i);
        }
    }
}

template <typename Packer>
inline void msgpack_pack(Packer& pk) const {
    pk.pack_array(3 + this->rows()*this->cols());
    pk.pack(std::string("__eigen__"));
    pk.pack(this->rows());
    pk.pack(this->cols());

    for (int i = 0; i < this->cols(); ++i) {
        for (int j = 0; j < this->rows(); ++j) {
            pk.pack(this->operator()(j, i));
        }
    }
}

template <typename MSGPACK_OBJECT>
inline void msgpack_object(MSGPACK_OBJECT* o, msgpack::zone* z) const { }

#endif  // MUNDY_ALENS_COMPUTE_MOBILITY_PERIPHERY_UTILS_EIGEN_MATRIX_PLUGIN_H_
