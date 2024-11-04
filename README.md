

---

# Movie Recommendation System Using Clustering and Reinforcement Learning

This project is a **Movie Recommendation System** that utilizes **KMeans clustering** for grouping movies based on genre and IMDb score. The recommendations are provided through **Q-learning** and **Policy Gradient** reinforcement learning methods to simulate dynamic and personalized recommendations based on user preferences.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project is designed to explore clustering techniques and reinforcement learning methods for recommending movies. Here, movies are clustered based on their genre and IMDb scores. Based on user interaction and feedback, two recommendation strategies, **Q-Learning** and **Policy Gradient**, are used to recommend movies dynamically within a selected mood or cluster.

## Features
- **KMeans Clustering**: Groups movies based on genre and IMDb scores.
- **Q-Learning Recommender**: Uses Q-learning to adapt movie recommendations based on simulated feedback.
- **Policy Gradient Recommender**: Recommends movies probabilistically, creating varied user experiences.
- **Visualization**: Visualizes the clustered data using PCA for dimensionality reduction.

## Setup and Installation

### Prerequisites
- Python 3.7 or higher
- Required libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/movie-recommendation-system.git
   cd movie-recommendation-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add movie dataset:**
   - Place your `movies_data.csv` file in the root directory.

## Usage

1. **Run the Script:**
   ```bash
   python main.py
   ```

2. **Select Cluster for Recommendations:**
   - The program will prompt you to enter a cluster number to get movie recommendations based on that cluster's mood.

3. **Review Recommendations:**
   - After selecting a cluster, recommendations will be generated based on both **Q-Learning** and **Policy Gradient** methods.

4. **Visualize Clustering Results:**
   - A plot will appear showing the KMeans clustering using PCA-reduced components.

### Sample Output
```plaintext
Enter cluster number (0 to N) to generate recommendations: 2
Generating recommendations for mood: Action
Q-Learning Recommended Movies for Action:
    Movie         Genre     IMDb score
0  Movie A      Action        7.5
1  Movie B      Action        8.1
...

Policy Gradient Recommended Movies for Action:
    Movie         Genre     IMDb score
0  Movie X      Action        6.9
1  Movie Y      Action        7.2
...
```

## Project Structure
```
├── README.md               # Project documentation
├── main.py                 # Main Python script to run the recommendation system
├── requirements.txt        # Python dependencies
├── movies_data.csv         # Movie dataset file
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request.

