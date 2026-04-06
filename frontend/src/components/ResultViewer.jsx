import './ResultViewer.css';

const COLOR_MAP = [
  { name: 'Trees', color: 'rgb(34, 139, 34)' },
  { name: 'Lush Bushes', color: 'rgb(0, 200, 0)' },
  { name: 'Dry Grass', color: 'rgb(210, 180, 140)' },
  { name: 'Dry Bushes', color: 'rgb(139, 90, 43)' },
  { name: 'Ground Clutter', color: 'rgb(128, 128, 0)' },
  { name: 'Flowers', color: 'rgb(255, 215, 0)' },
  { name: 'Logs', color: 'rgb(139, 69, 19)' },
  { name: 'Rocks', color: 'rgb(128, 128, 128)' },
  { name: 'Landscape', color: 'rgb(160, 82, 45)' },
  { name: 'Sky', color: 'rgb(135, 206, 235)' }
];

export default function ResultViewer({ resultImage, isPredicting, originalImage }) {
  return (
    <div className="result-container glass-panel">
      <h2 className="panel-title">2. Analysis Result</h2>
      
      <div className={`result-display ${!resultImage && !isPredicting ? 'empty' : ''}`}>
        
        {!originalImage && !isPredicting && !resultImage && (
          <div className="placeholder-state">
            <div className="icon-pulse">
               <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 16 16 12 12 8"></polyline><line x1="8" y1="12" x2="16" y2="12"></line></svg>
            </div>
            <p>Upload an image to see predictions</p>
          </div>
        )}

        {originalImage && !isPredicting && !resultImage && (
          <div className="placeholder-state ready">
           <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--accent-primary)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 16 16 12 12 8"></polyline><line x1="8" y1="12" x2="16" y2="12"></line></svg>
            <p>Ready for analysis. Click "Analyze Image" to start.</p>
          </div>
        )}

        {isPredicting && (
          <div className="processing-state">
            <div className="scan-line-container">
              {originalImage && <img src={originalImage.url} alt="Analyzing" className="scanning-image" />}
              <div className="scan-line"></div>
            </div>
            <p className="processing-text">Analyzing image features...</p>
          </div>
        )}

        {resultImage && !isPredicting && (
          <div className="result-image-wrapper fade-in">
            {/* The result image would be rendered here */}
            <img src={resultImage} alt="Model Result" className="final-result" />
            
            <div className="result-overlay">
              <span className="badge success">Analysis Complete</span>
            </div>
          </div>
        )}

      </div>
      
      {resultImage && (
        <div className="result-actions fade-in">
           <div className="color-legend">
             <span className="legend-title">Legend Reference:</span>
             <div className="color-pills">
               {COLOR_MAP.map(item => (
                 <div key={item.name} className="color-pill">
                    <span className="color-swatch" style={{ backgroundColor: item.color }}></span>
                    <span className="color-name">{item.name}</span>
                 </div>
               ))}
             </div>
           </div>

           <button className="action-btn secondary" onClick={() => {
              const link = document.createElement('a');
              link.href = resultImage;
              link.download = 'prediction_result.png';
              document.body.appendChild(link);
              link.click();
              document.body.removeChild(link);
           }}>
             <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
             Download Mask
           </button>
        </div>
      )}
    </div>
  );
}
