import os
import sys

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU founded: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("ERROR: GPU not found.")
    sys.exit(1)

mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)

IMAGE_NAME = "example.jpg"
IMAGE_PATH = "C:/Users/..." + IMAGE_NAME


def get_image_data(image_path):
    if not os.path.exists(image_path):
        print(f"Archive not found: {image_path}")
        sys.exit(1)

    img_pil = Image.open(image_path).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    return img_pil, img_tensor


def normalize(t):
    return (t - mean) / std

def apply_and_save_hires(original_pil, best_solution, original_path):
    W_orig, H_orig = original_pil.size
    num_pixels = len(best_solution) // 5

    final_img = original_pil.copy()

    print(f"\nApplying attack of 1 pixel to original resolution ({W_orig}x{H_orig})...")

    for p in range(num_pixels):
        idx = p * 5
        x_small = best_solution[idx]
        y_small = best_solution[idx + 1]
        r = best_solution[idx + 2]
        g = best_solution[idx + 3]
        b = best_solution[idx + 4]

        x_orig = int(round(x_small * (W_orig / 224.0)))
        y_orig = int(round(y_small * (H_orig / 224.0)))

        r_int = int(r * 255)
        g_int = int(g * 255)
        b_int = int(b * 255)

        x_orig = min(max(x_orig, 0), W_orig - 1)
        y_orig = min(max(y_orig, 0), H_orig - 1)

        final_img.putpixel((x_orig, y_orig), (r_int, g_int, b_int))

        print(f"   -> Pixel {p + 1}: Pos({x_orig}, {y_orig}) Cor RGB({r_int}, {g_int}, {b_int})")

    folder, filename = os.path.split(original_path)
    name, _ = os.path.splitext(filename)
    new_name = f"{name}_attacked.png"
    save_path = os.path.join(folder, new_name)

    final_img.save(IMAGE_PATH + "_attacked.png", format="png")
    print(f"Image saved: {save_path}")

def perturb_image_vectorized(xs, img_base):
    if xs.ndim < 2: xs = xs.unsqueeze(0)

    batch_size = xs.shape[0]
    imgs = img_base.repeat(batch_size, 1, 1, 1)
    _, _, H, W = imgs.shape
    num_pixels = xs.shape[1] // 5

    batch_indices = torch.arange(batch_size, device=device)

    for p in range(num_pixels):
        start = p * 5
        x = xs[:, start + 0].long().clamp(0, W - 1)
        y = xs[:, start + 1].long().clamp(0, H - 1)
        r = xs[:, start + 2]
        g = xs[:, start + 3]
        b = xs[:, start + 4]

        imgs[batch_indices, 0, y, x] = r
        imgs[batch_indices, 1, y, x] = g
        imgs[batch_indices, 2, y, x] = b

    return imgs


def run_attack_params(model, img_raw, target_class, pixels=1, maxiter=100, popsize=400):
    bounds_min = torch.tensor([0, 0, 0, 0, 0] * pixels, device=device).float()
    bounds_max = torch.tensor([224, 224, 1, 1, 1] * pixels, device=device).float()
    diff = bounds_max - bounds_min

    population = torch.rand(popsize, 5 * pixels, device=device)
    population = bounds_min + population * diff

    def get_fitness(pop):
        imgs = perturb_image_vectorized(pop, img_raw)
        with torch.no_grad():
            out = model(normalize(imgs))
            probs = F.softmax(out, dim=1)
        return probs[:, target_class]

    print(f"Starting attack (Pixels: {pixels}, Pop: {popsize})...")
    fitness = get_fitness(population)

    for i in range(maxiter):
        idxs_a = torch.randint(0, popsize, (popsize,), device=device)
        idxs_b = torch.randint(0, popsize, (popsize,), device=device)
        idxs_c = torch.randint(0, popsize, (popsize,), device=device)

        mutant = population[idxs_a] + 0.5 * (population[idxs_b] - population[idxs_c])
        mutant = torch.max(torch.min(mutant, bounds_max), bounds_min)

        cross_mask = torch.rand(popsize, 5 * pixels, device=device) < 0.7
        trial = torch.where(cross_mask, mutant, population)

        fitness_trial = get_fitness(trial)
        improve_mask = fitness_trial < fitness
        population[improve_mask] = trial[improve_mask]
        fitness[improve_mask] = fitness_trial[improve_mask]

        best_idx = torch.argmin(fitness)
        best_conf = fitness[best_idx].item()

        if (i + 1) % 10 == 0:
            print(f"   [Generation {i + 1}/{maxiter}] Original trust: {best_conf:.2%}")

        if best_conf < 0.5:
            best_sol = population[best_idx]
            # Validação
            final_img = perturb_image_vectorized(best_sol.unsqueeze(0), img_raw)
            pred = model(normalize(final_img)).argmax(1).item()

            if pred != target_class:
                print(f"Success! Changed to {pred}")
                return best_sol.cpu().numpy()

    print("Max generation achieved.")
    best_idx = torch.argmin(fitness)
    return population[best_idx].cpu().numpy()


if __name__ == "__main__":
    PIXELS = 1

    print("Loading model ResNet18...")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()

    img_pil_hires, img_tensor_lowres = get_image_data(IMAGE_PATH)

    with torch.no_grad():
        initial_pred = model(normalize(img_tensor_lowres)).argmax(1).item()
    print(f"Original detected: {initial_pred}")

    best_solution_vector = run_attack_params(
        model,
        img_tensor_lowres,
        initial_pred,
        pixels=PIXELS,
        maxiter=100,
        popsize=400
    )

    apply_and_save_hires(img_pil_hires, best_solution_vector, IMAGE_PATH)