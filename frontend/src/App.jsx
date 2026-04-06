import { useState } from 'react'
import './App.css'
import ImageUploader from './components/ImageUploader'
import ResultViewer from './components/ResultViewer'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [isPredicting, setIsPredicting] = useState(false)
  const [resultImage, setResultImage] = useState(null)

  const handleImageSelect = (file) => {
    const imageUrl = URL.createObjectURL(file)
    setSelectedImage({ file, url: imageUrl })
    setResultImage(null)
  }

  const handlePredict = async () => {
    if (!selectedImage) return;
    setIsPredicting(true);
    
    try {
      const formData = new FormData();
      formData.append('image', selectedImage.file);

      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to predict. Check python console.");
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setResultImage(imageUrl);
    } catch (error) {
      console.error(error);
      alert("Error predicting image. Ensure the Python backend is running.");
    } finally {
      setIsPredicting(false);
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>AI Vision Engine</h1>
        <p>Advanced image classification and precise segmentation</p>
      </header>

      <main className="main-content">
        <ImageUploader 
           onSelect={handleImageSelect} 
           selectedImage={selectedImage}
           onPredict={handlePredict}
           isPredicting={isPredicting}
        />
        <ResultViewer 
           resultImage={resultImage} 
           isPredicting={isPredicting}
           originalImage={selectedImage}
        />
      </main>
    </div>
  )
}

export default App
