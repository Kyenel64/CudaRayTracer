if not exist bin/ mkdir bin
nvcc -I./classes/ main.cu -o bin/app.exe
bin\app.exe