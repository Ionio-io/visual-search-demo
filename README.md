# VisualSearch: AI-Powered Visual Search Engine

VisionSearch is an intelligent visual search application that leverages Google's SigLIP model to find similar images through advanced embeddings and cosine similarity. Built with FastAPI and modern web technologies, it offers a seamless API for image similarity search.

![image](https://github.com/user-attachments/assets/52a9bb44-1301-48e6-9e0f-e35caad87beb)

---

## ✨ Features  

- **Image Similarity Search**: Upload an image to find visually similar results from your collection  
- **Image Database Management**: Add new images to your search database via API  
- **High Performance**: Utilizes Google's SigLIP-SO400M model with CUDA acceleration  
- **RESTful API**: Fully documented endpoints for easy integration   
- **Fast & Scalable**: Optimized pipeline for quick visual search results  
- **Responsive UI**: Clean, intuitive interface built with modern web frameworks

---

## 🛠 Tech Stack  

**Frontend**:  
- React + TypeScript  
- Vite (Next-gen frontend tooling)  
- shadcn-ui (Beautiful, accessible components)  
- Tailwind CSS (Utility-first styling)  

**Backend**:  
- FastAPI (Python backend service)  
- Google SigLIP-SO400M (State-of-the-art vision-language model)  
- PyTorch (Deep learning framework)  
- CUDA (GPU acceleration when available) 

**DevOps**:  
- GitHub Codespaces (Cloud development environments)  
- Netlify/Vercel (Deployment hosting)

---

## 🚀 Getting Started  

### Prerequisites  
- Python 3.10+  
- NVIDIA GPU with CUDA support (recommended)  
- pip package manager  
- NodeJS
- npm

---

### Getting Started Locally   
```bash
# Step 1: Clone the repository. 
git clone https://github.com/Ionio-io/vision-search-demo.git
cd vision-search

# Step 2: Create and activate virtual environment.
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate     # Windows

# Step 3: Install dependecies.
pip install -r requirements.txt
```

---

## Running the API
```bash
uvicorn main:app --reload
```
Access the API at: http://localhost:8000

---

## Running the frontend
```bash
npm run dev
```
This will be accessable at http://localhost:8080

---

## 📡API Endpoints 

-POST /query_image
-Find similar images to your query

Request:
Form-data with file field containing an image
Response:
```json
{
  "results": [
    {
      "path": "string",
      "similarity": float
    }
  ]
}
```

---

-POST /add_image
-Add new image to search database

Request:
Form-data with file field containing an image
Response:
```json
{
  "message": "string",
  "status_code": 200
}
```

---

## 🔮 Future Enhancements 

- Support for custom model fine-tuning
- Mobile app integration
- Multi-modal search (text + image)
- User accounts and search history

---

## 🤝 Contributing

We welcome contributions! Please:
- Fork the repository
- Create a feature branch
- Submit a pull request

---

##  License

This project is licensed under the Apache 2.0 License. See the LICENSE file for details.
