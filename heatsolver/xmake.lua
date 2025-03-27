add_rules("plugin.compile_commands.autoupdate", { outputdir = "./build" })
add_rules("utils.install.cmake_importfiles")
add_rules("mode.debug", "mode.release")

package("osqp")
do
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
end
package_end()

-- add_requires("osqp", { system = false, configs = { static } })
-- add_requires("abseil", { system = false })
-- add_requires("eigen", { system = false })

add_requires("fmt", { system = false })
add_requires("openmp", { system = false, configs = { ... } })

target("heatsolver")
do
	set_kind("$(kind)")

	add_headerfiles("./include/(**.h)", { install = true, public = true, prefixdir = "" })
	add_includedirs("./include/", { public = true })
	add_files("./src/view.cpp", "./src/solver.cpp", "./src/system.cpp", "./src/adjoint_solver.cpp")

	add_headerfiles("./thirdparty/osqp-cpp/include/(**.h)", { install = true, public = true, prefixdir = "" })
	add_includedirs("./thirdparty/osqp-cpp/include/", { public = true })

	-- add_files("./src/inverse_solver.cpp")
	-- add_files("./thirdparty/osqp-cpp/src/osqp++.cc")

	add_packages("fmt", "openmp", { public = true })
	-- add_packages("osqp", "abseil", "eigen", { public = true })
end
target_end()
