import React, { useState, useRef } from 'react';
import "../styles/home.css";

function DragDropImageUploader() {
  const [mediaFiles, setMediaFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedDetectors, setSelectedDetectors] = useState([]);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const fileInputRef = useRef(null);

  async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];

    if (!file) {
      alert('Please select a file!');
      return;
    }

    // Use dynamic models based on selected detectors
    const models = selectedDetectors;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('models', JSON.stringify(models));

    try {
      const response = await fetch('http://localhost:8000/models/', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        console.log(result);
      } else {
        console.error('Error uploading file:', response.statusText);
      }
    } catch (error) {
      console.error('Error uploading file:', error.message);
    }
  }

  function selectFiles() {
    fileInputRef.current.click();
  }

  function onMediaSelect(event) {
    const files = event.target.files;
    if (files.length === 0) return;

    for (let i = 0; i < files.length; i++) {
      if (!isValidFile(files[i])) continue;

      if (!mediaFiles.some((e) => e.name === files[i].name)) {
        setMediaFiles((prevMediaFiles) => [
          ...prevMediaFiles,
          {
            name: files[i].name,
            type: files[i].type,
            file: files[i],
            url: URL.createObjectURL(files[i]),
          },
        ]);
      }
    }
  }

  function isValidFile(file) {
    const supportedTypes = ['image', 'video'];
    const fileType = file.type.split('/')[0];
    return supportedTypes.includes(fileType);
  }

  function deleteMedia(index) {
    setMediaFiles((prevMediaFiles) => prevMediaFiles.filter((_, i) => i !== index));
  }

  function onDragOver(event) {
    event.preventDefault();
    setIsDragging(true);
    event.dataTransfer.dropEffect = 'copy';
  }

  function onDragLeave(event) {
    event.preventDefault();
    setIsDragging(false);
  }

  function onDrop(event) {
    event.preventDefault();
    setIsDragging(false);
    const files = event.dataTransfer.files;

    for (let i = 0; i < files.length; i++) {
      if (!isValidFile(files[i])) continue;

      if (!mediaFiles.some((e) => e.name === files[i].name)) {
        setMediaFiles((prevMediaFiles) => [
          ...prevMediaFiles,
          {
            name: files[i].name,
            type: files[i].type,
            file: files[i],
            url: URL.createObjectURL(files[i]),
          },
        ]);
      }
    }
  }

  function handleCheckboxChange(event) {
    const { value, checked } = event.target;

    if (checked) {
      setSelectedDetectors((prevDetectors) => [...prevDetectors, value]);
    } else {
      setSelectedDetectors((prevDetectors) =>
        prevDetectors.filter((detector) => detector !== value)
      );
    }
  }

  function toggleDarkMode() {
    setIsDarkMode(!isDarkMode);
  }

  return (
    <div className={`app-container ${isDarkMode ? 'dark-mode' : ''}`}>
      <nav className="navbar">
        <div className="navbar-buttons">
          <div className="dark-mode-button">
            <label className="switch">
              <input type="checkbox" checked={isDarkMode} onChange={toggleDarkMode} />
              <span className="slider round">
                {isDarkMode ? <i className="fas fa-moon"></i> : <i className="fas fa-sun"></i>}
              </span>
            </label>
          </div>
          <button className="nav-button">Login</button>
          <button className="nav-button">Profile</button>
        </div>
      </nav>

      <div className="content-container">
        <div className="left-half">
          <h1>The Detection statistics come here</h1>
          <p>Graphs will be displayed here</p>
        </div>
        <div className="right-half">
          <div className="card">
            <div className="top">
              <p>Drag and Drop media uploading</p>
            </div>
            <div
              className={`drag-area ${isDragging ? 'drag-over' : ''}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
            >
              {isDragging ? (
                <span className="select">Drop media here</span>
              ) : (
                <>
                  Drag & Drop media here or{' '}
                  <span className="select" role="button" onClick={selectFiles}>
                    Browse
                  </span>
                </>
              )}
              <input
                id="fileInput"
                name="file"
                type="file"
                className="file"
                multiple
                ref={fileInputRef}
                onChange={onMediaSelect}
              />
            </div>
            <div className="container">
              {mediaFiles.map((media, index) => (
                <div className="media" key={index}>
                  <span className="delete" onClick={() => deleteMedia(index)}>
                    &times;
                  </span>
                  {media.type.startsWith('image') ? (
                    <img src={media.url} alt={media.name} />
                  ) : (
                    <video width="100%" height="100%" controls>
                      <source src={media.url} type={media.type} />
                      Your browser does not support the video tag.
                    </video>
                  )}
                </div>
              ))}
            </div>
            <div className="checkbox-container">
              {["Detector 1", "Detector 2", "Detector 3", "Detector 4"].map((detectorValue, index) => (
                <label key={index} className="checkbox-label">
                  <input
                    type="checkbox"
                    value={detectorValue}
                    onChange={handleCheckboxChange}
                    checked={selectedDetectors.includes(detectorValue)}
                  />
                  <span className="custom-checkbox"></span>
                  {detectorValue}
                </label>
              ))}
            </div>
            <button type="button" onClick={uploadFile}>
              Upload File
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DragDropImageUploader;
