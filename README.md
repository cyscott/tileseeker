# TileSeeker 🍼

OCR tool for finding your child's name on the St. Joseph's Hospital baby name tile wall.

## About

St. Joseph's Hospital has a beautiful tradition of creating name tiles for children born there. These tiles, displayed on a commemorative wall, contain the child's name and date of birth. **TileSeeker** helps parents search through photos of these tile walls to find their children's names using AI-powered optical character recognition.

This project was born from a parent's desire to locate their children's names among thousands of tiles - turning what could be hours of manual searching into a quick, automated process.

## Features

- 📸 Processes high-resolution images of the tile wall
- 🔍 Intelligent image chunking to handle large photos
- 🤖 Uses OpenAI's GPT-4 Vision for accurate OCR
- 📍 Provides exact location information (which part of the image)
- 📊 Exports results to CSV and JSON formats
- 🔄 Handles multiple images in batch processing

## Prerequisites

- Python 3.7+
- OpenAI API key with GPT-4 Vision access
- Photos of the St. Joseph's baby name tile wall

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/stjoe-names.git
cd stjoe-names
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
cp .env.example .env
```

Then edit `.env` and add your actual OpenAI API key:
```
OPENAI_API_KEY=your_actual_api_key_here
```

## Usage

### Basic Search

Place your tile wall photos in the `./images` directory, then search for a name:

```bash
python ocr_name_search.py "Charlotte"
```

### Search Options

```bash
# Search for a name in a specific image
python ocr_name_search.py "Emily" --single-image ./images/wall_section_5.jpeg

# Search in a different directory
python ocr_name_search.py "Michael" --images-dir /path/to/photos

# Export only to CSV
python ocr_name_search.py "Sophia" --export csv

# Enable debug mode to see API responses
python ocr_name_search.py "James" --debug

# Adjust parallel processing (default: 4)
python ocr_name_search.py "Oliver" --max-workers 2
```

### Output

The tool provides:
- **Terminal output**: Shows all matches with name, date of birth, and location
- **CSV export**: `name_search_[name]_[timestamp].csv`
- **JSON export**: `name_search_[name]_[timestamp].json`

Each result includes:
- Full name as it appears on the tile
- Date of birth
- Source image filename
- Location within the image (e.g., "top-left", "middle-center")
- Exact pixel coordinates

## How It Works

1. **Image Splitting**: Large photos are intelligently divided into overlapping chunks that stay within OpenAI's processing limits
2. **Parallel Processing**: Multiple chunks are processed simultaneously for faster results
3. **Smart Matching**: Case-insensitive partial name matching
4. **Deduplication**: Removes duplicate findings from overlapping chunks

## Tips for Best Results

- Take photos with good lighting and minimal glare
- Ensure tiles are clearly visible and in focus
- Higher resolution images yield better results
- The tool works best with photos taken straight-on (minimal angle)

## Privacy Note

This tool processes images locally and only sends them to OpenAI's API for OCR. No data is stored or shared beyond your API calls.

## Contributing

Feel free to submit issues or pull requests if you have suggestions for improvements!

## License

MIT License - feel free to use this tool to find your little one's tile! 👶
