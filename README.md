# Mosquito Sentinel: AI-Powered Mosquito Species Detection

Mosquito Sentinel is an innovative AI-driven application that detects mosquito species in real-time using their unique wingbeat sounds. Built with deep learning and deployed via a user-friendly Streamlit web app, it identifies 38 mosquito species, provides critical information on associated diseases and control measures, and lays the foundation for crowdsourced risk mapping to prevent mosquito-borne diseases like malaria, dengue, and Zika. This project tackles a global health crisis with accessible, scalable technology, making it ideal for communities, researchers, and public health officials.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features
- **Real-Time Detection**: Records audio via microphone and identifies mosquito species using a pre-trained CNN model.
- **Species Information**: Displays danger level, diseases spread (e.g., malaria, dengue), and tailored control solutions (e.g., repellents, larvicides).
- **Noise-Robust**: Trained on mixed audio (mosquito + environmental noise) for reliable real-world performance.
- **User-Friendly Interface**: Streamlit app with start/stop buttons, animated detection display, and clear information panels.
- **Scalable Foundation**: Designed for future integration of geolocation-based risk mapping and community-driven data collection.
- **Open-Source Data**: Utilizes the HumBugDB dataset and ESC-50 noise dataset for transparency and reproducibility.

## Installation
Follow these steps to set up Mosquito Sentinel locally.

### Prerequisites
- Python 3.8+
- Git
- A microphone-enabled device
- Internet connection (for initial data download)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/mosquito-sentinel.git
   cd mosquito-sentinel
   ```

2. **Install Dependencies**:
   Create a virtual environment and install required packages:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

   Sample `requirements.txt`:
   ```
   torch==2.0.1
   torchaudio==2.0.2
   sounddevice==0.4.6
   streamlit==1.29.0
   numpy==1.24.3
   pandas==2.0.3
   librosa==0.10.1
   soundfile==0.12.1
   scikit-learn==1.3.0
   tqdm==4.66.1
   ```

3. **Download Datasets**:
   - Mosquito audio: Clone the HumBugDB repository and download audio files from Zenodo:
     ```bash
     git clone https://github.com/HumBug-Mosquito/HumBugDB.git
     wget https://zenodo.org/record/4904800/files/humbugdb_neurips_2021_1.zip?download=1
     wget https://zenodo.org/record/4904800/files/humbugdb_neurips_2021_2.zip?download=1
     wget https://zenodo.org/record/4904800/files/humbugdb_neurips_2021_3.zip?download=1
     wget https://zenodo.org/record/4904800/files/humbugdb_neurips_2021_4.zip?download=1
     unzip humbugdb_neurips_2021_*.zip -d HumBugDB/data/audio
     ```
   - Noise data: Download the ESC-50 dataset or use your own environmental noise files.

4. **Set Up Paths**:
   Update paths in the Jupyter notebook and `app.py` to point to your local dataset directories:
   - Mosquito CSV: `HumBugDB/data/metadata/neurips_2021_zenodo_0_0_1.csv`
   - Noise CSV: Path to your noise metadata (e.g., `esc50_noise.csv`)
   - Audio directories: Adjust `mosquito_audio_dir` and `noise_audio_dir`

5. **Download Pre-Trained Model**:
   Place the pre-trained model (`mosquito_cnn_final_torchscript.pt`) in the project root or train your own using the provided Jupyter notebook.

## Usage
1. **Train the Model** (Optional):
   - Open the Jupyter notebook (`train_mosquito_cnn.ipynb`).
   - Update paths to your audio and metadata files.
   - Run all cells to preprocess data, train the CNN, and export the model as TorchScript.

2. **Run the App**:
   ```bash
   streamlit run app.py
   ```
   - Open the provided URL (e.g., `http://localhost:8501`) in your browser.
   - Click "Start Listening" to begin real-time audio capture.
   - View detected species, confidence, danger level, diseases, and control measures.
   - Click "Stop Listening" to pause detection.

3. **Example Output**:
   - Detection: "Detected: ae aegypti (confidence: 0.92) 🦟"
   - Info:
     ```
     Species Information
     Danger Level: HIGH
     Diseases Spread: Dengue, Zika, Chikungunya, Yellow Fever
     Solutions to Control:
     - Eliminate standing water in containers
     - Use repellents (DEET, picaridin)
     - Insecticide-treated nets
     ```

## Project Structure
```
mosquito-sentinel/
│
├── app.py                    # Streamlit app for real-time detection
├── train_mosquito_cnn.ipynb  # Jupyter notebook for data prep and model training
├── mosquito_cnn_final_torchscript.pt  # Pre-trained model
├── requirements.txt          # Python dependencies
├── HumBugDB/                 # Cloned dataset repository
│   ├── data/
│   │   ├── audio/            # Mosquito audio files
│   │   ├── metadata/         # Metadata CSV
├── noise_data/               # Environmental noise files (e.g., ESC-50)
├── README.md                 # This file
```

## Technical Details
- **Data Pipeline**:
  - Audio: 16kHz, mono, 2s clips (padded/trimmed).
  - Processing: Mel spectrograms (128 mels, 1024 FFT, 160 hop length), augmented with SpecAugment and Gaussian noise.
  - Dataset: Mixed mosquito + noise (positive) and pure noise (negative) samples, balanced with weighted sampling.

- **Model**:
  - Architecture: TinyCNN with 4 ConvBlocks (1→32→64→128→256 channels), ~500k parameters.
  - Input: Mel spectrogram (1x128xT).
  - Output: 39 classes (38 mosquito species + "no mosquito").
  - Training: CrossEntropy loss, Adam optimizer, early stopping, CPU-only.

- **Deployment**:
  - Streamlit app with Torchaudio for audio processing and sounddevice for capture.
  - Inference: Converts 1s audio to 64x64 Mel spectrogram, predicts via TorchScript model.

- **Performance**:
  - Validation accuracy: 85-95% (varies by dataset quality).
  - Inference time: <100ms on CPU, suitable for real-time use.

## Future Work
- **Crowdsourcing**: Add geolocation to user detections for real-time risk mapping.
- **Outbreak Prediction**: Use ML to forecast outbreaks based on species density and environmental factors.
- **Mobile Integration**: Deploy as iOS/Android apps for wider accessibility.
- **Enhanced Features**: Add visual mosquito ID, multi-language support, or federated learning for privacy.
- **Public Health Impact**: Partner with NGOs and governments for targeted interventions (e.g., larvicide campaigns).

## Contributing
We welcome contributions! To get started:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## Acknowledgements
- **HumBugDB**: For providing the mosquito audio dataset (Zenodo: 4904800).
- **ESC-50**: For environmental noise data.
- **Streamlit, PyTorch, Torchaudio**: For enabling rapid development and deployment.
- **WHO & CDC**: For public health data inspiring this project.

For issues or questions, contact Akash BR at akashbellaryramesh123@gmail.com. Let's fight mosquito-borne diseases together! 🦟
