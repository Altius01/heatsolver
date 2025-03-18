add_rules("plugin.compile_commands.autoupdate", { outputdir = "./build" })
add_rules("utils.install.cmake_importfiles")
add_rules("mode.debug", "mode.release")

package("osqp")
set_sourcedir(path.join(os.scriptdir(), "thirdparty/osqp/"))
on_install(function(package)
  local configs = {}
  table.insert(configs, "-DCMAKE_BUILD_TYPE=" .. (package:debug() and "Debug" or "Release"))
  table.insert(configs, "-DBUILD_SHARED_LIBS=" .. (package:config("shared") and "ON" or "OFF"))
  import("package.tools.cmake").install(package, configs)
end)
on_test(function(package)
  assert(package:has_cfuncs("csc_matrix", { includes = "osqp/cs.h" }))
end)
package_end()
add_requires("osqp")

add_requires("abseil")
add_requires("fmt")
add_requires("eigen")

target("heatsolver")
do
  set_kind("$(kind)")

  add_headerfiles("./include/(**.h)", { install = true, public = true, prefixdir = "" })
  add_includedirs("./include/", { public = true })
  add_files("./src/*.cpp")

  add_headerfiles("./thirdparty/osqp-cpp/include/(**.h)", { install = true, public = true, prefixdir = "" })
  add_includedirs("./thirdparty/osqp-cpp/include/", { public = true })
  add_files("./thirdparty/osqp-cpp/src/osqp++.cc")

  add_packages("fmt", "osqp", "abseil", "eigen")
end
target_end()
