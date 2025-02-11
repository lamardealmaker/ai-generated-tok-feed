try:
    import moviepy.editor as mp
    print("MoviePy imported successfully!")
except Exception as e:
    print(f"Error importing MoviePy: {str(e)}")
    import sys
    print(f"Python path: {sys.path}")
