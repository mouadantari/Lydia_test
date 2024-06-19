# Overview of the Repository Structure

- src/: Contains the source code written in python for the application.
  - \_\_init\_\_.py: Marks the directory as a package.
  - api.py: Defines the API routes.
  - embeddings.py: Contains code for generating embeddings.
  - inference.py: Contains code for retrieving the closest images.
  - test.py: Unit tests for various functions in the repository.
  - utils.py: Utility functions.
  - explore.ipynb: A notebook containing a quick data science oriented analysis of the solution.
- .dockerignore: Specifies files to be ignored by Docker.
- .gitignore: Specifies files to be ignored by Git.
- Dockerfile: Builds a Docker image to encapsulate the application.
- Makefile: Simplifies building the Docker image, running a container, testing the code, requesting the API, and stopping the container.
- data.csv: CSV file containing the dataset.
- image-downloader.py: Python script for downloading images from the links in the CSV.
- requirements.txt: Lists dependencies to be installed.

# Usage

## Prerequisites

To be able to use the application, please download and install [Docker](https://www.docker.com/products/docker-desktop/), [make](https://www.gnu.org/software/make/#download) and [GIT](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) if any of them is not installed in your machine

## Usage of the application

### Clone the repository

To clone the repository use:

```
git clone https://github.com/mouadantari/Lydia_test.git
```

After cloning the repository, change the working directory to the root of the cloned repository:

```
cd <path_to_repository>
```

### Build the docker image

To run the application for the very first time you need at first to build the image encapsulating the app using:

```
make build
```

This command is meant to be run only once in each machine.

### Run tests (optional)

To run the unit test implemented in the app use:

```
make test
```

### Run the application

After the docker image is built, to run the application use:

```
make run
```

### Request the POST endpoint of the API

While the application is running, to find the IDs of the closest image use

```
make request IMAGE_PATH=<absolute_path_of_image>
```

Replace <absolute_path_of_image> with the absolute path of your image.

### Stop the application

After you finish using the app, you can stop it using:

```
make stop
```

You'll need to restart the application with `make run` to use it again.

# Approach

The ImageFinder solution aims to identify the closest matching images from a dataset based on a given input image. This approach leverages deep learning techniques for image embedding generation and a simple API to interact with these embeddings to find similar images.

## Comparing images

With the variety of images that the dataset contains, the solution necessitates a robust method for feature extraction to handle diverse image content effectively. That's why I chose ResNet50V2 to generate embeddings.

This choice is based on its balance between performance and computational efficiency. Also to quantify the difference between two images, the usage of cosine distance make the task simple and easily interpretable since the distances stays withing the same scale.

Embeddings are generated then stored in a CSV file. This allows quick access to the embeddings without recomputing them each time we want to search for an image, especially with the limited size of the used dataset.
But it may not be the most efficient for very large datasets. In that case, it might become necessary to use more scalable storage options such as a database optimized for high-dimensional data.

## API

Then to interact with the solution, I implemented an API with only 2 routes: a GET route `/ping` that simply returns `"pong"` for checks (in case the api is plugged with an automatic system), and a POST route `/closest_img_ids` that takes in input an image, then processes it, generates its embedding, and returns the IDs of the closest images by comparing the embedding of the input to the stored embeddings using the cosine distance.

## Unit tests

I also included some unit tests to ensure the reliability of the solution. I only tested some functions where some behavior is expected. We could push further the tests by including the API in the tested parts, as well as some aspects of embedding generation like the shape of the generated embedding. Those tests would be more useful if included in a CI/CD pipeline to ensure the robustness of the framework in case of new changes or improvements introduced.

## Docker usage

Then I used docker to encapsulate the solution in an image ready to be used to run a container for the application. This guarantees that the application runs consistently regardless of the underlying infrastructure. Also since the application is meant to accept new inputs, the usage of a mounted volume was necessary to ensure this functionality.

## Makefile usage

The usage of a Makefile makes the usage of the application easier and more straightforward, from building the image, running a container for the app, testing the code, and requesting the api. For the last point, the makefile reduces constraints on the users side, by allowing them to use any photo within any local path instead of constraining them to put it in the mounted directory.

## Potential Improvements

As potential improvements of this solution, we can site the handling of large datasets when some parts of the solution will be hardly scalable, especially the storage part, and the computation of distances between a new embedding and old ones. Also in a real life situation, a CI/CD pipeline is necessary to ensure fluid deployment in case of changes and improvements as well as a more exhaustive testing strategy.

Also depending on the exact usage of the solution a more robust approach can be used. For example for a face matching solution, we might need in this case a fine tuning or transfer learning on a dataset with human faces to better capture human face features, as well as an adapted metric like precision or f-beta scores with beta<1.
