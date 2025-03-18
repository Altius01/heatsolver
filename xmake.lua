add_rules("plugin.compile_commands.autoupdate", { outputdir = "./build" })
add_rules("utils.install.cmake_importfiles")
add_rules("mode.debug", "mode.release")

includes("./heatsolver")

add_requires("conan::highfive/2.10.0", { alias = "highfive" })

target("application")
do
  set_kind("binary")

  add_headerfiles("./app/include/(**.h)", { install = true, prefixdir = "" })
  add_includedirs("./app/include/", { public = true })
  add_files("./app/src/*.cpp")

  add_deps("heatsolver")
  add_packages("highfive")
end
target_end()
