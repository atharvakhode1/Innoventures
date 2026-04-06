import { useRef, useState } from 'react';
import './ImageUploader.css';

export default function ImageUploader({ onSelect, selectedImage, onPredict, isPredicting }) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      onSelect(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      onSelect(e.target.files[0]);
    }
  };

  return (
    <div className="uploader-container glass-panel">
      <h2 className="panel-title">1. Upload Input</h2>
      
      {!selectedImage ? (
        <div 
          className={`drop-zone ${isDragging ? 'dragging' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current.click()}
        >
          <div className="upload-icon">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
          </div>
          <p className="drop-text">Drag and drop an image here</p>
          <p className="drop-subtext">or click to browse from your computer</p>
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleFileChange}
            accept="image/*"
            style={{ display: 'none' }}
          />
        </div>
      ) : (
        <div className="preview-container">
          <div className="image-wrapper">
             <img src={selectedImage.url} alt="Selected" className="preview-image" />
             <button 
               className="change-image-btn"
               onClick={() => fileInputRef.current.click()}
             >
               Change File
             </button>
             <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileChange}
              accept="image/*"
              style={{ display: 'none' }}
            />
          </div>
          
          <button 
            className={`predict-btn ${isPredicting ? 'loading' : ''}`}
            onClick={onPredict}
            disabled={isPredicting}
          >
            {isPredicting ? (
              <>
                <span className="spinner"></span>
                Processing Model...
              </>
            ) : (
              'Analyze Image'
            )}
          </button>
        </div>
      )}
    </div>
  );
}
