import cv2
import pytesseract
import numpy as np

# Configure the path to Tesseract if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Sudoku solver
def solve_sudoku(board):
    def find_empty(board):
        for r in range(6):
            for c in range(6):
                if board[r][c] == 0:
                    return (r, c)
        return None
    def is_valid(board, num, pos):
        row, col = pos
        if any(board[row][c] == num for c in range(6) if c != col):
            return False
        if any(board[r][col] == num for r in range(6) if r != row):
            return False
        box_row_start, box_col_start = (row // 2)*2, (col // 3)*3
        for r in range(box_row_start, box_row_start+2):
            for c in range(box_col_start, box_col_start+3):
                if board[r][c] == num and (r, c) != pos:
                    return False
        return True
    find = find_empty(board)
    if not find:
        return True
    row, col = find
    for num in range(1, 7):
        if is_valid(board, num, (row, col)):
            board[row][col] = num
            if solve_sudoku(board):
                return True
            board[row][col] = 0
    return False

def find_sudoku_box(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_area, best_box = 0, None
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # Look for rectangular shapes
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area, best_box = area, approx
    if best_box is None:
        raise Exception("Sudoku puzzle not found!")
    pts = best_box.reshape(4,2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect = np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)]
    ])
    side = int(max([
        np.linalg.norm(rect[0]-rect[1]),
        np.linalg.norm(rect[1]-rect[2]),
        np.linalg.norm(rect[2]-rect[3]),
        np.linalg.norm(rect[3]-rect[0])
    ]))
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(image, M, (side, side))
    return warp

def parse_grid_from_image(puzzle_img):
    puzzle_img = cv2.resize(puzzle_img, (300, 300))
    cell_h = cell_w = 50
    result = np.zeros((6, 6), dtype=int)
    gray = cv2.cvtColor(puzzle_img, cv2.COLOR_BGR2GRAY)
    for i in range(6):
        for j in range(6):
            x, y = j*cell_w, i*cell_h
            cell_img = gray[y:y+cell_h, x:x+cell_w]
            cell_crop = cell_img[8:42, 8:42]
            cell_crop = cv2.resize(cell_crop, (28,28))
            cell_crop = cv2.GaussianBlur(cell_crop, (3,3), 0)
            cell_crop = cv2.adaptiveThreshold(cell_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
            digit = pytesseract.image_to_string(
                cell_crop, 
                config='--psm 10 -c tessedit_char_whitelist=123456'
            )
            digit = ''.join(filter(str.isdigit, digit))
            result[i][j] = int(digit) if digit else 0
    return result, puzzle_img

def output_solution_image(board, puzzle_img):
    img = puzzle_img.copy()
    cell_h = cell_w = 50
    for i in range(6):
        for j in range(6):
            x, y = j*cell_w + 11, i*cell_h + 38
            cv2.putText(img, str(board[i][j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 128, 255), 2, cv2.LINE_AA)
    return img

if __name__ == "__main__":
    input_file = "C:/Users/udayv/Desktop/puzzle.jpg"  # Change to your screenshot filename if needed
    original = cv2.imread(input_file)
    cropped = find_sudoku_box(original)
    grid, norm_puzzle = parse_grid_from_image(cropped)
    print("Parsed grid:\n", grid)
    if solve_sudoku(grid):
        print("Solved grid:\n", grid)
        solution_img = output_solution_image(grid, norm_puzzle)
        cv2.imwrite('sudoku_solution.jpg', solution_img)
        print("Solved image saved as sudoku_solution.jpg")
    else:
        print("No solution found!")
