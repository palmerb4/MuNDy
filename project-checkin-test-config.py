# This file allows project-level configuration of the checkin-test system to
# set project options that are required for all developers. Machine or package
# specific options should not be placed in this file.

# This is a dictionary that specifies project-specific options for the
# checkin-test script that should be used by all developers. This
# includes default command line arguments that must be passed to
# checkin-test as well as settings for specific builds.

configuration = {

    # The default command line arguments that should be used by all developers.
    'defaults': {
        '--send-email-to-on-push': 'trilinos-checkin-tests@software.sandia.gov',
        '--no-rebase' : '',
        },

    # CMake options (-DVAR:TYPE=VAL) cache variables.
    'cmake': {
        
        # Options that are common to all builds.
        'common': [],

        # Defines --default-builds, in order.
        'default-builds': [
            # Options for the MPI_DEBUG build.
            ('MPI_DEBUG', [
                '-DTPL_ENABLE_MPI:BOOL=ON',
                '-DCMAKE_BUILD_TYPE:STRING=RELEASE',
                '-DTribitsExProj_ENABLE_DEBUG:BOOL=ON',
                '-DTribitsExProj_ENABLE_CHECKED_STL:BOOL=ON',
                '-DTribitsExProj_ENABLE_DEBUG_SYMBOLS:BOOL=ON',
                ]),
            # Options for the SERIAL_RELEASE build.
            ('SERIAL_RELEASE', [
                '-DTPL_ENABLE_MPI:BOOL=OFF',
                '-DCMAKE_BUILD_TYPE:STRING=RELEASE',
                '-DTribitsExProj_ENABLE_DEBUG:BOOL=OFF',
                '-DTribitsExProj_ENABLE_CHECKED_STL:BOOL=OFF',
                ]),
            ], # default-builds

        }, # cmake

    } # configuration
