#  6×6 Sudoku Solver (Image → Solution)

An end-to-end **6×6 Sudoku Solver** that detects a Sudoku puzzle from an image, extracts the grid using **OCR**, solves it using **backtracking**, and displays the solution through a modern **web UI**.

This project combines **Computer Vision**, **OCR**, **algorithmic problem solving**, and a **frontend–backend architecture**.

---

##  Features

-  Upload an image of a **6×6 Sudoku puzzle**
-  Automatically detects and crops the Sudoku grid
-  Extracts digits using **Tesseract OCR**
-  Solves the puzzle using a **backtracking algorithm**
-  Allows manual editing of detected values before solving
-  Generates a solution image with answers overlaid
-  Interactive UI built with **HTML + Tailwind CSS**
-  REST API powered by **Flask (Python)**

---

##  Project Structure

```text
.
├── puzzle.py                # Core image processing + Sudoku solver logic
├── sudoku_solver.html       # Frontend UI
├── sudoku_solution.jpg      # Generated output image (after solving)
└── README.md
