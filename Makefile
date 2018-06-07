CC          := g++

# internal flags
CCFLAGS   := -O3 -fopenmp 
LDFLAGS     := -lfftw3f -lgomp

# Target rules
all: build 

build: obj rectv

obj/main.o:src/main.cpp
	$(CC) $(CCFLAGS) -o $@ -c $<
obj/rectv.o:src/rectv.cpp
	$(CC) $(CCFLAGS) -o $@ -c $<
obj/radonusfft.o:src/radonusfft.cpp
	$(CC) $(CCFLAGS) -o $@ -c $<
obj:
	mkdir -p obj
rectv: obj/main.o obj/rectv.o obj/radonusfft.o
	$(CC) -o $@ $+ $(LDFLAGS)
clean:
	rm -rf rectv obj

