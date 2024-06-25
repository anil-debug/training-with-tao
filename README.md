# Ollama with docker

## Introduction
This guide will help you set up and run the Ollama Docker container in both CPU-only and GPU-accelerated modes using Docker Compose.

## 1. Run the service
### Start the cpu only service
```
docker-compose up -d ollama-cpu
```
### Start the GPU-accelerated service:
```
docker-compose up -d ollama-gpu
```

## 2. Downloading the Model
### For the CPU-only version:
```
docker exec -it ollama ollama pull llama3
```
### For the GPU-accelerated version:
```
docker exec -it ollama-gpu ollama pull llama3
```
## 3. List and run the model
### For the CPU-only version:
```
docker exec -it ollama ollama list
docker exec -it ollama ollama run llama3

```
### For the GPU-accelerated version:
```
docker exec -it ollama-gpu ollama list
docker exec -it ollama-gpu ollama run llama3
```
