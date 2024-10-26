
# 📸 imageTextify

imageTextify is a powerful and lightweight image-to-text recognition tool designed for extracting textual content from images using advanced OCR (Optical Character Recognition) technology. It leverages the power of **PaddleOCR** and **FastAPI** for efficient backend processing, making it a perfect solution for developers looking to integrate OCR capabilities into their projects.

![imageTextify Banner](https://github.com/Udoy2/ImageTextify_Server/blob/main/assets/banner.png)

## 🌟 Features

- 🔍 **Accurate OCR**: Uses PaddleOCR for high-quality text recognition.
- 🚀 **Fast and Lightweight**: Optimized with Python 3.11-slim for efficient performance.
- 🌐 **Web-based Interface**: Simple UI to upload images and get OCR results.
- 📦 **Dockerized Deployment**: Easy to deploy with Docker.

## 🚀 Quick Start

### Prerequisites

- Docker
- Python 3.11+
- Pip

### Docker Deployment

To quickly deploy the app using Docker:

```bash
docker build -t imagetextify .
docker run -p 8000:8000 imagetextify
```

### Local Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Udoy2/ImageTextify_Server.git
   cd ImageTextify_Server
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App**

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

4. Visit the app at `http://localhost:8000`.

## 🐳 Docker Setup

This project uses a lightweight Python 3.11-slim Docker image. To build and run the container:

```bash
docker build -t imagetextify .
docker run -p 8000:8000 imagetextify
```

## 🛠️ Development

### File Structure

- `main.py`: FastAPI server code.
- `requirements.txt`: Contains all Python dependencies.
- `Dockerfile`: Docker configuration for building the image.

### Adding More Dependencies

If you need to add more dependencies:

1. Add them to `requirements.txt`.
2. Rebuild the Docker container:

   ```bash
   docker build -t imagetextify .
   ```

## 📄 API Endpoints

### Upload Image

**Endpoint**: `/uploadImage`  
**Method**: `POST`  
**Description**: Upload an image to extract text.

**Example**:

```bash
curl -X POST -F 'file=@your-image.jpg' http://localhost:8000/uploadImage
```

## 📋 License

This project is licensed under the MIT License.

## 🤝 Contributing

Feel free to fork this repository, make improvements, and send a pull request. Contributions are welcome!

## 🌐 Connect

- **Website**: [mahmudrahman.me](https://mahmudrahman.me)
- **Email**: udoyrahman983@gmail.com

