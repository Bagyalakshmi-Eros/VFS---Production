# VFS 


## Prerequisites

- Python 3.11
- Git

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Immerso-AIIP-Ltd/Vfs-Proxy.git
cd Vfs-Proxy
```

### 2. Create Virtual Environment

```bash
python3.11 -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Navigate to Backend Directory

```bash
cd backend
```

### 4. Install Requirements

```bash
pip install -r requirements.txt
```

### 5. Install PyTorch (CUDA Support)

```bash
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

### 6. Download Required Assets

From the [Releases](https://github.com/Immerso-AIIP-Ltd/Vfs-Proxy/releases) section, download the following folders:

- **weights** - Place this folder inside the `face-swap` directory
- **uploaded-videos** - Place this folder in the root directory of the project

**Directory Structure After Setup:**
```
Vfs-Proxy/
├── backend/
│   └── requirements.txt
├── face-swap/
│   └── weights/          # Downloaded from releases     
└── main.py
```

## Running the Application

After completing all the setup steps above, run the application using uvicorn:

```bash
uvicorn main:app --reload
```

The application will be available at `http://localhost:8000`

## Notes

- Ensure you have CUDA-compatible GPU for optimal performance
- Make sure all dependencies are installed before running the application
- The virtual environment should remain activated while running the application
