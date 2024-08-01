[app]
# (str) Title of your application
title = Face Detection App

# (str) Package name
package.name = facedetection

# (str) Package domain (needed for android/ios packaging)
package.domain = org.example

# (str) Source code directory (relative to the spec file)
source.dir = .

# (list) Source files to be included (comma-separated)
source.include_exts = py,png,jpg,kv,atlas

# (list) Application requirements (comma-separated)
requirements = python3,kivy,opencv-python

# (str) Application entry point
# The entry point should be the name of the Python file without the `.py` extension.
entrypoint = main

# (list) Permissions
android.permissions = INTERNET, CAMERA

# (str) Orientation (one of 'landscape', 'portrait', 'all')
orientation = portrait

# (str) Full name including package path of the Java class that implements Python Activity
# For most applications, the default is fine.
android.entrypoint = org.kivy.android.PythonActivity

# (str) Android SDK directory (required if ANDROID_SDK_ROOT is not set)
android.sdk_path = C:/Users/Rinki/AppData/Local/Android/Sdk

# (str) Android NDK directory (required if ANDROID_NDK_ROOT is not set)
android.ndk_path = C:/Program Files/Android/Android Studio/plugins/android-ndk/lib

# (str) Application version
version = 0.1

# (list) Screensize compatibility
android.screen_sizes = small, normal, large, xlarge

# (str) Presplash image filename (optional)
# presplash.filename =

# (str) Presplash color (optional)
# presplash.color =

# (str) Icon filename (optional)
# icon.filename =

# (list) Meta-data to be included in the manifest (optional)
# android.meta_data =

# (str) Additional Java .jar files to add (optional)
# android.add_jars =

# (str) Java .jar files to remove (optional)
# android.remove_jars =

# (str) Activities to add (optional)
# android.add_activities =

# (list) Directories to add to the PATH environment variable (optional)
# android.add_path =

# (str) Services to add (optional)
# android.add_services =

# (list) Java classes to add (optional)
# android.add_classes =

# (list) Additional permissions (optional)
# android.add_permissions =

# (str) Java source files to add (optional)
# android.add_src_files =

# (str) Build configuration file (optional)
# android.config =

# (str) Java .so files to add (optional)
# android.add_so =

# (str) Java .dex files to add (optional)
# android.add_dex =

# (str) Additional C++ .so files to add (optional)
# android.add_cpp =

# (str) Additional python-for-android recipes (optional)
# android.add_recipes =

# (str) Java .so files to remove (optional)
# android.remove_so =

# (str) Directories to add to the CLASSPATH environment variable (optional)
# android.add_classpath =

# (str) Additional environments (optional)
# android.add_envs =

# (str) Directories to add to the PYTHONPATH environment variable (optional)
# android.add_pythonpath =

# (list) Shared libraries to add (optional)
# android.add_libs =

# (str) Shared libraries to remove (optional)
# android.remove_libs =

# (list) Shared libraries to include (optional)
# android.add_shared_libs =

# (str) Shared libraries to remove (optional)
# android.remove_shared_libs =

# (list) Additional files to include in the APK (optional)
# android.add_files =

# (list) Additional files to remove from the APK (optional)
# android.remove_files =

# (list) Additional directories to include in the APK (optional)
# android.add_dirs =

# (str) Additional directories to remove from the APK (optional)
# android.remove_dirs =

# (list) Additional environment variables to set (optional)
# android.add_env_vars =

# (list) Additional Java classes to use (optional)
# android.add_java_classes =

# (list) Additional Java methods to use (optional)
# android.add_java_methods =

# (list) Additional Java constants to use (optional)
# android.add_java_constants =

# (list) Additional resources to add to the APK (optional)
# android.add_res =

# (list) Additional resources to remove from the APK (optional)
# android.remove_res =

# (list) Additional asset directories to add to the APK (optional)
# android.add_assets =

# (list) Additional asset directories to remove from the APK (optional)
# android.remove_assets =

# (list) Additional binary files to include in the APK (optional)
# android.add_bins =

# (list) Additional binary files to remove from the APK (optional)
# android.remove_bins =

# (str) Directory in which python-for-android should look for the target's build cache
# (optional, you usually don't need this)
# android.p4a_cache_dir =

# (list) Key-value pairs to set in the Android manifest (optional)
# android.add_manifest_attrs =

# (list) Key-value pairs to set in the Android resource xml file (optional)
# android.add_res_attrs =

# (str) Additional Android manifest file to include (optional)
# android.add_manifest =

# (str) Additional Android resource xml file to include (optional)
# android.add_res =

# (str) Additional Android asset directory to include (optional)
# android.add_asset =

# (str) Additional Android binary file to include (optional)
# android.add_bin =

# (str) Additional environment variables to set during the build (optional)
# android.add_env_vars_build =

# (str) Additional environment variables to set during the package phase (optional)
# android.add_env_vars_package =

# (list) Additional environment variables to set during the final package phase (optional)
# android.add_env_vars_final =

# (str) Android API level to use (optional)
# android.api =

# (str) Target platform for the build (optional, default: 'android')
target = android

# (str) List of Android ABIs to build for (optional, default: 'armeabi-v7a')
android.archs = armeabi-v7a

# (str) The version code to use for the APK (optional)
# android.version_code =

# (str) The version name to use for the APK (optional)
# android.version_name =

# (str) Application icon filename (optional)
# icon.filename =

# (str) Application presplash filename (optional)
# presplash.filename =

# (str) Presplash color (optional)
# presplash.color =

# (str) Splash screen filename (optional)
# splash.filename =

# (str) Splash screen color (optional)
# splash.color =

# (str) Additional key-value pairs to set in the Android manifest (optional)
# android.add_manifest_attrs =

# (str) Additional attributes to set in the Android resource xml file (optional)
# android.add_res_attrs =

# (str) Full path to a Java .jar file to add (optional)
# android.add_jar =

# (str) Full path to a Java .jar file to remove (optional)
# android.remove_jar =

# (str) Full path to a shared library .so file to add (optional)
# android.add_shared_lib =

# (str) Full path to a shared library .so file to remove (optional)
# android.remove_shared_lib =

# (list) Additional shared libraries to add (optional)
# android.add_libs =

# (list) Additional shared libraries to remove (optional)
# android.remove_libs =

# (list) Additional binaries to include in the APK (optional)
# android.add_bins =

# (list) Additional binaries to remove from the APK (optional)
# android.remove_bins =

# (str) Directory in which python-for-android should look for recipes (optional)
# android.p4a_dir =

# (str) Directory in which python-for-android should look for the target's build cache (optional)
# android.p4a_cache_dir =

# (str) Additional Java class path (optional)
# android.add_classpath =

# (str) Additional Java .so files to include (optional)
# android.add_so =

# (str) Additional Java .so files to remove (optional)
# android.remove_so =

# (str) Additional environment variables to set (optional)
# android.add_env_vars =

# (str) Additional directories to include in the APK (optional)
# android.add_dirs =

# (str) Additional directories to remove from the APK (optional)
# android.remove_dirs =

# (list) Additional files to include in the APK (optional)
# android.add_files =

# (list) Additional files to remove from the APK (optional)
# android.remove_files =

# (list) Additional environment variables to set

[buildozer]
# (int) Log level (0 = error only, 1 = warning, 2 = info, 3 = debug, 4 = trace)
log_level = 2

# (int) Display warning if buildozer is running as root (0 = False, 1 = True)
warn_on_root = 1

# (str) Path to the directory in which Buildozer should look for the target's build cache
# build_cache =

# (str) Path to the directory in which Buildozer should look for recipes
# buildozer.p4a_dir =

# (str) Path to the directory in which Buildozer should look for the target's build cache
# buildozer.p4a_cache_dir =

# (bool) Copy the tar files into the versionned directory before packaging
# buildozer.copy_tars = 0
