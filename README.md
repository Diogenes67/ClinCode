# 🏥 ClinCode

Professional AI-powered medical diagnosis coding system for healthcare providers.

## Overview

ClinCode is a streamlined application that helps healthcare professionals quickly and accurately code clinical notes using ICD-10 classifications. The system uses advanced AI to analyze medical text and suggest appropriate diagnosis codes with confidence scores.

## Features

- 🔍 **Real-time Classification**: Instant analysis of clinical notes
- 📊 **Confidence Scoring**: Each diagnosis comes with a confidence percentage
- 📋 **ICD-10 Integration**: Standard medical coding system
- 💾 **Export Options**: Download results as CSV or PDF
- 🎨 **Professional Interface**: Clean, medical-grade UI design
- 🔒 **Secure Processing**: All data processed locally

## Installation

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/Diogenes67/ClinCode.git
cd ClinCode
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from your forked repository

## Usage

1. **Enter Clinical Note**: Type or paste the clinical note in the text area
2. **Click "Classify Note"**: The AI will analyze the text
3. **Review Results**: Check the suggested ICD-10 codes and confidence scores
4. **Export**: Download results as CSV for your records

## Technology Stack

- **Frontend**: Streamlit
- **AI/ML**: TensorFlow, PyTorch, Transformers
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly

## Requirements

- Python 3.8+
- 4GB RAM minimum
- Internet connection (for model downloads)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions, please open an issue on GitHub.

---

**Note**: This tool is designed to assist healthcare professionals and should not replace professional medical judgment.
