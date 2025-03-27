add_rules("plugin.compile_commands.autoupdate", { outputdir = "./build" })
add_rules("utils.install.cmake_importfiles")
add_rules("mode.debug", "mode.release")

set_config("kind", "shared")

includes("./heatsolver")

add_requires("highfive", { system = false })
add_requires("boost >= 1.83.0", { system = false })

target("application")
do
	set_kind("binary")

	add_headerfiles("./app/include/(**.h)", { install = true, prefixdir = "" })
	add_includedirs("./app/include/", { public = true })
	add_files("./app/src/*.cpp")

	add_deps("heatsolver")
	add_packages("highfive", "boost")
end
target_end()
