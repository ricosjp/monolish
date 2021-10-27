How to generate document
-------------------------

Documents in this directory use variables defined in cmake.
You have to run `doxygen` command after cmake replaces these variables.
cmake defines a target for running `doxygen`:

```
cd /path/to/monolish/repo/top/
cmake -B build .                        # Generates build/doc/* files
cmake --build build/ --target document  # Run doxygen at build/
```

If you want to check generated markdown files, you can find them at `build/doc/`.
