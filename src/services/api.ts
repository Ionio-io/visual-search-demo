/**
 * API client for interacting with the FastAPI backend
 */

export interface SearchResult {
  path: string;
  similarity: number;
}

export interface SearchResponse {
  results: SearchResult[];
}

const API_URL = 'http://localhost:8000';

/**
 * Search for similar images using the query_image endpoint
 */
export async function searchSimilarImages(file: File): Promise<SearchResult[]> {
  const formData = new FormData();
  formData.append('file', file);

  console.log("Sending POST request to /query_image with file:", file);

  const response = await fetch(`${API_URL}/query_image`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    console.error("Error response from /query_image:", errorData);
    throw new Error(errorData.error || 'Failed to search image');
  }

  const data: SearchResponse = await response.json();
  console.log("Received response from /query_image:", data);
  return data.results;
}

/**
 * Add image to the default_images folder
 */
export async function addImage(file: File): Promise<string> {
  const formData = new FormData();
  formData.append('file', file);

  console.log("Sending POST request to /add_image with file:", file);

  const response = await fetch(`${API_URL}/add_image`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json();
    console.error("Error response from /add_image:", errorData);
    throw new Error(errorData.detail || 'Failed to add image');
  }

  const data = await response.json();
  console.log("Received response from /add_image:", data);
  return data.message;
}