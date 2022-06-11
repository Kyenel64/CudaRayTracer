if not exist bin/ mkdir bin
nvcc -I./classes/ -I./vendor/ main.cu -o bin/app.exe
bin\app.exe