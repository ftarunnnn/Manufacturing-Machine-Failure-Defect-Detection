import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

def create_synthetic_images(images_per_class=100, image_size=(128, 128), random_seed=42):
    np.random.seed(random_seed)
    base_dir = os.path.join('data', 'raw', 'images')
    classes = ['normal', 'crack', 'scratch', 'dent', 'surface_defect']
    
    for c in classes:
        os.makedirs(os.path.join(base_dir, c), exist_ok=True)
        
    print(f"[INFO] Generating synthetic product images for classes: {classes}")
    
    for c in classes:
        for i in range(images_per_class):
            # Base metallic brushed texture background
            base = np.random.normal(160, 15, (image_size[1], image_size[0], 3)).astype(np.float32)
            # Add subtle metallic horizontal grain
            grain = np.tile(np.random.uniform(-10, 10, (image_size[1], 1, 3)), (1, image_size[0], 1))
            img_arr = np.clip(base + grain, 0, 255).astype(np.uint8)
            
            img = Image.fromarray(img_arr)
            draw = ImageDraw.Draw(img)
            
            if c == 'normal':
                # Clean surface with minor natural noise
                pass
                
            elif c == 'crack':
                # Jagged dark fracture line across surface
                start_x = np.random.randint(20, 40)
                start_y = np.random.randint(20, 40)
                curr_x, curr_y = start_x, start_y
                points = [(curr_x, curr_y)]
                for _ in range(8):
                    curr_x += np.random.randint(8, 15)
                    curr_y += np.random.randint(6, 14) * np.random.choice([-1, 1])
                    curr_y = np.clip(curr_y, 10, image_size[1] - 10)
                    points.append((curr_x, curr_y))
                draw.line(points, fill=(30, 30, 30), width=np.random.randint(2, 4))
                
            elif c == 'scratch':
                # Linear bright/dark parallel scratch streaks
                for _ in range(np.random.randint(2, 5)):
                    x1 = np.random.randint(10, image_size[0] - 40)
                    y1 = np.random.randint(10, image_size[1] - 40)
                    length = np.random.randint(30, 60)
                    angle = np.random.uniform(0, np.pi)
                    x2 = int(x1 + length * np.cos(angle))
                    y2 = int(y1 + length * np.sin(angle))
                    draw.line([(x1, y1), (x2, y2)], fill=(230, 230, 230), width=np.random.randint(1, 3))
                    draw.line([(x1+1, y1+1), (x2+1, y2+1)], fill=(50, 50, 50), width=1)
                    
            elif c == 'dent':
                # Circular shadow gradient simulating surface indentation
                cx = np.random.randint(30, image_size[0] - 30)
                cy = np.random.randint(30, image_size[1] - 30)
                radius = np.random.randint(12, 25)
                # Outer shadow
                draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(90, 90, 95))
                # Inner highlight
                draw.ellipse([cx - radius + 4, cy - radius + 4, cx + radius - 2, cy + radius - 2], fill=(190, 190, 195))
                img = img.filter(ImageFilter.GaussianBlur(radius=2))
                
            elif c == 'surface_defect':
                # Patchy oxidation / irregular discoloration spots
                draw_arr = np.array(img).astype(np.float32)
                for _ in range(np.random.randint(3, 7)):
                    sx = np.random.randint(20, image_size[0] - 30)
                    sy = np.random.randint(20, image_size[1] - 30)
                    rw = np.random.randint(15, 35)
                    rh = np.random.randint(15, 35)
                    draw_arr[sy:sy+rh, sx:sx+rw, 0] += np.random.uniform(30, 60)  # Red rust tone
                    draw_arr[sy:sy+rh, sx:sx+rw, 1] -= np.random.uniform(20, 40)
                    draw_arr[sy:sy+rh, sx:sx+rw, 2] -= np.random.uniform(20, 40)
                img_arr = np.clip(draw_arr, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_arr)
                
            file_name = f"{c}_{i+1:03d}.png"
            file_path = os.path.join(base_dir, c, file_name)
            img.save(file_path)
            
    print(f"[SUCCESS] Synthetic image dataset created in {base_dir}")
    print(f"Total Images: {images_per_class * len(classes)} (100 per class)")

if __name__ == '__main__':
    create_synthetic_images()
