# @HEADER
# **********************************************************************************************************************
#
# Mundy: Multi-body Nonlocal Dynamics
# Copyright 2023 Flatiron Institute
# Author: Bryce Palmer
#
# Mundy is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
#
# Mundy is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with Mundy. If not, see
# <https://www.gnu.org/licenses/>.
#
# **********************************************************************************************************************
# @HEADER

# Specialized logic for what file changes should trigger a global build in CI
# testing where testing should only occur package impacted by the change.

class ProjectCiFileChangeLogic:

  def isGlobalBuildFileRequiringGlobalRebuild(self, modifiedFileFullPath):
    modifiedFileFullPathArray = modifiedFileFullPath.split('/')
    lenPathArray = len(modifiedFileFullPathArray)
    if lenPathArray==1:
      # Files sitting directly under <projectDir>/
      if modifiedFileFullPathArray[0] == "CMakeLists.txt":
        return True
      if modifiedFileFullPathArray[0].rfind(".cmake") != -1:
        return True
    elif modifiedFileFullPathArray[0] == 'cmake':
      # Files under <projectDir>/cmake/
      if modifiedFileFullPathArray[1]=='ExtraRepositoriesList.cmake':
        return False
      elif modifiedFileFullPathArray[1] == 'ctest' and lenPathArray >= 3:
        if lenPathArray > 3:
          # This is a file
          # <projectDir>/cmake/ctest/<something>/[...something...]  so this is
          # for a specific machine and should not trigger a global build.
          return False
        else:
          # Any other file directly under cmake/ctest/ should trigger a global
          # build.
          return True
      else:
        # All other files under cmake/
        if modifiedFileFullPath.rfind(".cmake") != -1:
          # All other *.cmake files under cmake/ trigger a global build.
          return True
    # Any other files should not trigger a global build
    return False
