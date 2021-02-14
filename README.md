# computer_vision
People detection and tracking

# Assignment2 Computer Vision Course

This software was developed from Matteo Spadetto in order to detect and track people in a given video from MOT challenge.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

This software was implemented with `OpenCV` in `C++` so you basically need the source code, `cmake` and `make`. The `Makefile` is provided in the folder.

## Compile and run on VScode or linux-base terminal
To compile and run this software just type the following commands:

```
cd /path/to/folder/Assignment2
```
```
cmake .
```
```
make
```
```
./cv_assignment_2
```
Inside the folder, the executable file `./cv_assignment_2` already exists so you can skip the second and third steps (you need them only after you make changes on source code).

In the souce code the path to the frame files are written for linux based systems. If you have another operative system you might have to change the path accordingly with your system rules.
```
    String folderpath = "./Video/img1/";
```

## Built With

- [VScode](https://code.visualstudio.com/) - VScode
- [OpenCV](https://opencv.org/) - OpenCV C++

## Versioning

This is the first version of `Assignment2 Computer Vision Course`. Probably a new version will be deployed later with new improvments.

## Matteo Spadetto contacts

For issues and questions please contact me at matteo.spadetto-1@studenti.unitn.it

- **Output video video_out.avi** - [video](https://drive.google.com/drive/folders/1BZFmSzgSjBEi6md7Nv_Oqsl5EgFYTasr?usp=sharing)
- **Matteo Spadetto GitHub** - [eagletrt](https://github.com/MatteoSpadetto)
