#!/usr/bin/env python3
"""
OCR Name Search Script
Splits large images into smaller chunks to process with OpenAI API
and searches for specific names on hospital name tiles.
"""

import base64
import os
import sys
from PIL import Image
import math
from openai import OpenAI
import json
from typing import List, Dict, Tuple, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import csv
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with API key from environment
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Constants
PATCH_SIZE = 32  # Size of each patch in pixels
MAX_PATCHES = 1536  # Maximum number of patches allowed by OpenAI
MAX_DIMENSION = int(math.sqrt(MAX_PATCHES * PATCH_SIZE * PATCH_SIZE))  # Max dimension for a single chunk


def encode_image(image_path: str) -> str:
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def calculate_chunk_dimensions(image_width: int, image_height: int) -> Tuple[int, int]:
    """
    Calculate optimal chunk dimensions to stay under the patch limit.
    Returns (chunk_width, chunk_height)
    """
    # Calculate total patches for the full image
    total_patches = (image_width * image_height) // (PATCH_SIZE * PATCH_SIZE)
    
    if total_patches <= MAX_PATCHES:
        # Image is small enough to process as-is
        return image_width, image_height
    
    # Need to split the image
    # Start with square chunks and adjust
    chunk_size = MAX_DIMENSION
    
    # Adjust chunk size to create reasonable overlaps
    while chunk_size > PATCH_SIZE * 10:  # Minimum chunk size
        num_chunks_x = math.ceil(image_width / chunk_size)
        num_chunks_y = math.ceil(image_height / chunk_size)
        
        # Add 10% overlap to avoid missing names at boundaries
        overlap = int(chunk_size * 0.1)
        effective_chunk_size = chunk_size - overlap
        
        if effective_chunk_size > 0:
            break
        
        chunk_size = int(chunk_size * 0.9)
    
    return chunk_size, chunk_size


def get_position_description(position: Tuple[int, int, int, int], image_width: int, image_height: int) -> str:
    """
    Generate a human-readable description of where in the image a chunk is located.
    """
    x, y, x_end, y_end = position
    
    # Calculate relative positions
    x_center = (x + x_end) / 2
    y_center = (y + y_end) / 2
    
    # Determine horizontal position
    if x_center < image_width * 0.33:
        h_pos = "left"
    elif x_center < image_width * 0.67:
        h_pos = "center"
    else:
        h_pos = "right"
    
    # Determine vertical position
    if y_center < image_height * 0.33:
        v_pos = "top"
    elif y_center < image_height * 0.67:
        v_pos = "middle"
    else:
        v_pos = "bottom"
    
    return f"{v_pos}-{h_pos}"


def split_image(image_path: str) -> List[Dict]:
    """
    Split image into chunks that fit within OpenAI's patch limit.
    Returns list of dictionaries with chunk info and base64 data.
    """
    img = Image.open(image_path)
    width, height = img.size
    
    print(f"Original image size: {width}x{height} pixels")
    print(f"Total patches if processed as-is: {(width * height) // (PATCH_SIZE * PATCH_SIZE)}")
    
    chunk_width, chunk_height = calculate_chunk_dimensions(width, height)
    
    chunks = []
    overlap = int(min(chunk_width, chunk_height) * 0.1)  # 10% overlap
    
    y = 0
    chunk_id = 0
    
    while y < height:
        x = 0
        while x < width:
            # Define chunk boundaries
            x_end = min(x + chunk_width, width)
            y_end = min(y + chunk_height, height)
            
            # Crop the chunk
            chunk = img.crop((x, y, x_end, y_end))
            
            # Save chunk temporarily and encode
            temp_path = f"temp_chunk_{chunk_id}.jpg"
            chunk.save(temp_path, "JPEG", quality=95)
            
            # Encode and store chunk info
            position = (x, y, x_end, y_end)
            chunks.append({
                'id': chunk_id,
                'position': position,
                'area_description': get_position_description(position, width, height),
                'base64': encode_image(temp_path),
                'filename': os.path.basename(image_path),
                'full_path': image_path,
                'image_dimensions': (width, height)
            })
            
            # Clean up temp file
            os.remove(temp_path)
            
            chunk_id += 1
            
            # Move to next chunk with overlap
            x += chunk_width - overlap
            if x + overlap >= width:
                break
        
        y += chunk_height - overlap
        if y + overlap >= height:
            break
    
    print(f"Split image into {len(chunks)} chunks of approximately {chunk_width}x{chunk_height} pixels each")
    return chunks


def process_chunk_for_names(chunk_data: Dict, search_name: str, debug: bool = False) -> List[Dict]:
    """
    Process a single image chunk with OpenAI to find names.
    Returns list of found names with their details.
    """
    try:
        prompt = f"""Please analyze this image section which contains hospital name tiles. 
        Each tile typically has a child's name and their date of birth.
        
        I'm specifically looking for names containing: "{search_name}"
        
        For EACH name tile you can see in this image section:
        1. Extract the full name
        2. Extract the date of birth (month, day, year)
        
        Return the results as a JSON array with objects containing:
        - "full_name": the complete name as shown
        - "date_of_birth": the date in format "Month Day, Year"
        - "contains_search": true if this name contains "{search_name}" (case-insensitive), false otherwise
        
        Even if a name doesn't match the search, include it in the results.
        If you can't read a tile clearly, skip it.
        If no names are visible, return an empty array: []
        
        Return ONLY the JSON array, no other text. Example:
        [{{"full_name": "John Smith", "date_of_birth": "January 15, 2023", "contains_search": true}}]"""
        
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{chunk_data['base64']}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        if debug:
            print(f"\nChunk {chunk_data['id']} raw response: {response_text[:200]}...")
        
        # Try to parse JSON response
        try:
            # Clean up common issues
            response_text = response_text.strip()
            
            # Remove markdown code block markers if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            names = json.loads(response_text)
            
            if not isinstance(names, list):
                if debug:
                    print(f"Warning: Response is not a list for chunk {chunk_data['id']}")
                names = []
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON from chunk {chunk_data['id']}: {e}")
            if debug:
                print(f"Response text: {response_text}")
            names = []
        
        # Add chunk information to each name
        for name in names:
            name['chunk_id'] = chunk_data['id']
            name['chunk_position'] = chunk_data['position']
            name['area_description'] = chunk_data['area_description']
            name['source_file'] = chunk_data['filename']
            name['full_path'] = chunk_data['full_path']
            name['image_dimensions'] = chunk_data['image_dimensions']
        
        return names
        
    except Exception as e:
        print(f"Error processing chunk {chunk_data['id']}: {str(e)}")
        return []


def search_names_in_image(image_path: str, search_name: str, max_workers: int = 4, debug: bool = False) -> List[Dict]:
    """
    Search for names in an image by splitting it into chunks and processing each.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return []
    
    print(f"\nProcessing image: {image_path}")
    print(f"Searching for: {search_name}")
    
    # Split image into chunks
    chunks = split_image(image_path)
    
    # Process chunks in parallel
    all_names = []
    matches = []
    
    print(f"\nProcessing {len(chunks)} chunks...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all chunks for processing
        future_to_chunk = {
            executor.submit(process_chunk_for_names, chunk, search_name, debug): chunk 
            for chunk in chunks
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(future_to_chunk)):
            chunk = future_to_chunk[future]
            try:
                names = future.result()
                all_names.extend(names)
                
                # Check for matches
                chunk_matches = [n for n in names if n.get('contains_search', False)]
                matches.extend(chunk_matches)
                
                print(f"  Chunk {chunk['id']}: Found {len(names)} names, {len(chunk_matches)} matches")
                
            except Exception as e:
                print(f"  Chunk {chunk['id']}: Failed - {str(e)}")
            
            # Small delay to avoid rate limiting
            if i < len(chunks) - 1:
                time.sleep(0.1)
    
    return matches, all_names


def export_results(matches: List[Dict], search_name: str, export_format: str = "both") -> str:
    """
    Export search results to CSV and/or JSON files.
    Returns the filenames created.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"name_search_{search_name.replace(' ', '_')}_{timestamp}"
    
    filenames = []
    
    if export_format in ["csv", "both"]:
        csv_filename = f"{base_filename}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'full_name', 'date_of_birth', 'source_file', 'full_path', 
                'area_description', 'chunk_position', 'image_dimensions'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for match in matches:
                row = {
                    'full_name': match.get('full_name', ''),
                    'date_of_birth': match.get('date_of_birth', ''),
                    'source_file': match.get('source_file', ''),
                    'full_path': match.get('full_path', ''),
                    'area_description': match.get('area_description', ''),
                    'chunk_position': str(match.get('chunk_position', '')),
                    'image_dimensions': str(match.get('image_dimensions', ''))
                }
                writer.writerow(row)
        filenames.append(csv_filename)
    
    if export_format in ["json", "both"]:
        json_filename = f"{base_filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(matches, jsonfile, indent=2, ensure_ascii=False)
        filenames.append(json_filename)
    
    return filenames


def search_all_images(search_name: str, images_dir: str = "./images", export_format: str = "both", debug: bool = False) -> None:
    """
    Search for a name across all images in the specified directory.
    """
    if not os.path.exists(images_dir):
        print(f"Error: Images directory not found: {images_dir}")
        return
    
    # Get all JPEG files
    image_files = [
        os.path.join(images_dir, f) 
        for f in os.listdir(images_dir) 
        if f.lower().endswith(('.jpg', '.jpeg'))
    ]
    
    if not image_files:
        print(f"No JPEG images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    all_matches = []
    
    # Process each image
    for image_path in image_files:
        matches, _ = search_names_in_image(image_path, search_name, debug=debug)
        all_matches.extend(matches)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"SEARCH RESULTS FOR: {search_name}")
    print(f"{'='*60}")
    
    if all_matches:
        print(f"\nFound {len(all_matches)} matches:\n")
        
        # Remove duplicates based on full_name and date_of_birth
        unique_matches = []
        seen = set()
        
        for match in all_matches:
            key = (match.get('full_name', ''), match.get('date_of_birth', ''))
            if key not in seen:
                seen.add(key)
                unique_matches.append(match)
        
        for i, match in enumerate(unique_matches, 1):
            print(f"{i}. Name: {match.get('full_name', 'Unknown')}")
            print(f"   Date of Birth: {match.get('date_of_birth', 'Unknown')}")
            print(f"   File: {match.get('source_file', 'Unknown')}")
            print(f"   Location: {match.get('area_description', 'Unknown')} area")
            print(f"   Pixel coordinates: {match.get('chunk_position', 'Unknown')}")
            print()
        
        # Export results
        if export_format != "none":
            exported_files = export_results(unique_matches, search_name, export_format)
            print(f"\nResults exported to: {', '.join(exported_files)}")
    else:
        print(f"\nNo matches found for '{search_name}'")


def main():
    parser = argparse.ArgumentParser(
        description="Search for names in hospital name tile images using OCR"
    )
    parser.add_argument(
        "search_name",
        help="Name to search for (partial matches supported)"
    )
    parser.add_argument(
        "--images-dir",
        default="./images",
        help="Directory containing JPEG images (default: ./images)"
    )
    parser.add_argument(
        "--single-image",
        help="Process only a specific image file"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of parallel API calls (default: 4)"
    )
    parser.add_argument(
        "--export",
        choices=["csv", "json", "both", "none"],
        default="both",
        help="Export format for results (default: both)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to see raw API responses"
    )
    
    args = parser.parse_args()
    
    if args.single_image:
        matches, _ = search_names_in_image(args.single_image, args.search_name, args.max_workers, args.debug)
        
        print(f"\n{'='*60}")
        print(f"SEARCH RESULTS FOR: {args.search_name}")
        print(f"{'='*60}")
        
        if matches:
            print(f"\nFound {len(matches)} matches:\n")
            for i, match in enumerate(matches, 1):
                print(f"{i}. Name: {match.get('full_name', 'Unknown')}")
                print(f"   Date of Birth: {match.get('date_of_birth', 'Unknown')}")
                print(f"   File: {match.get('source_file', 'Unknown')}")
                print(f"   Location: {match.get('area_description', 'Unknown')} area")
                print(f"   Pixel coordinates: {match.get('chunk_position', 'Unknown')}")
                print()
            
            # Export results
            if args.export != "none":
                exported_files = export_results(matches, args.search_name, args.export)
                print(f"\nResults exported to: {', '.join(exported_files)}")
        else:
            print(f"\nNo matches found for '{args.search_name}'")
    else:
        search_all_images(args.search_name, args.images_dir, args.export, args.debug)


if __name__ == "__main__":
    main()