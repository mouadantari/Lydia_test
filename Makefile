# Variables
IMAGE_NAME=image-search-api
CONTAINER_NAME=image-search-container
HOST_PORT=8000
CONTAINER_PORT=8000

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container
run:
	docker run --name $(CONTAINER_NAME) -p $(HOST_PORT):$(CONTAINER_PORT) -v $(PWD)/images_to_search:/app/images_to_search -d $(IMAGE_NAME)

# Stop and remove the Docker container
stop:
	docker stop $(CONTAINER_NAME)
	docker rm $(CONTAINER_NAME)

# Make an API request to find the 3 closest images
# Usage: make request IMAGE_PATH=path/to/image.jpg
request:
	cp $(IMAGE_PATH) $(PWD)/images_to_search/temp.jpg
	curl -X POST "http://localhost:$(HOST_PORT)/closest_img_ids" -H "Content-Type: application/json" -d '{"image_path": "images_to_search/temp.jpg"}'
	rm $(PWD)/images_to_search/temp.jpg

# Full workflow: build, run, and make a request
# Usage: make full IMAGE_PATH=path/to/image.jpg
full: build run request

.PHONY: build run stop request full
