"""Task 2: Primitive parameter otpimizaiton with RANSAC"""
import numpy as np
from scipy.optimize import least_squares
import trimesh


def create_cylinder_mesh(center, direction, radius, height, color=[0, 1, 0]):
    """
    Create a cylinder mesh in trimesh centered at `center` and aligned to `direction`.

    Args:
        center (np.ndarray): The center point of the cylinder.
        direction (np.ndarray): A vector indicating the cylinder's orientation.
        radius (float): The radius of the cylinder.
        height (float): The height of the cylinder.
        color (list): RGB color of the cylinder.

    Returns:
        trimesh.Trimesh: A trimesh object representing the cylinder.
    """
    # Create a cylinder aligned with the Z-axis
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=32)

    # Normalize the direction vector
    direction = np.array(direction)
    direction /= np.linalg.norm(direction)

    # Compute the rotation matrix to align the cylinder's Z-axis with the given direction vector
    z_axis = np.array([0, 0, 1])  # The default axis of the cylinder
    rotation_matrix = trimesh.geometry.align_vectors(z_axis, direction)

    # Apply rotation to the cylinder
    cylinder.apply_transform(rotation_matrix)

    # Translate the cylinder to the desired center position
    cylinder.apply_translation(center)

    # Apply color to the cylinder mesh
    cylinder.visual.face_colors = np.array(color + [1.0]) * 255  # Color the mesh faces

    return cylinder


def create_sphere_mesh(center, radius, color=[1, 0, 0]):
    """
    Create a sphere mesh in trimesh centered at `center`.

    Args:
        center (np.ndarray): The center of the sphere.
        radius (float): The radius of the sphere.
        color (list): RGB color of the sphere.

    Returns:
        trimesh.Trimesh: A trimesh object representing the sphere.
    """
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere.apply_translation(center)
    sphere.visual.face_colors = np.array(color + [1.0]) * 255  # Color the mesh faces

    return sphere

def sphere_residuals(params, points):
    """
    Compute the residuals for fitting points to a sphere.

    Args:
        params (np.array): A 1x4 array representing the sphere parameters [x0, y0, z0, r].
        points (np.array): Nx3 array of points to fit the sphere to.

    Returns:
        np.array: Residuals representing the difference between distances and the radius.
    """
    # Sphere parameters
    x0, y0, z0, r = params
    center = np.array([x0, y0, z0])

    ### Your code here ###
    # Compute the residuals: distance of each point to sphere surface
    distances = np.linalg.norm(points - center, axis=1)
    residual = distances - r
    ### End of your code ###
    return residual

def sphere_jac(params, points):
    """
    Compute the Jacobian matrix for fitting points to a sphere.

    Args:
        params (np.array): A 1x4 array representing the sphere parameters [x0, y0, z0, r].
        points (np.array): Nx3 array of points to fit the sphere to.

    Returns:
        np.array: Jacobian matrix (Nx4), with partial derivatives of the residuals with respect to [x, y, z, r].
    """
    # Sphere parameters
    x0, y0, z0, r = params
    center = np.array([x0, y0, z0])

    ### Your code here ###
    # Initialize the Jacobian matrix (Nx4)
    J = np.zeros((points.shape[0], 4))

    diff = center - points
    distances = np.linalg.norm(diff, axis=1)
    distances = np.maximum(distances, 1e-12) # avoid division by zero

    # Partial derivatives w.r.t x0, y0, z0 (center coordinates)
    J[:, 0] = diff[:, 0] / distances
    J[:, 1] = diff[:, 1] / distances
    J[:, 2] = diff[:, 2] / distances
    
    # Partial derivative w.r.t r (radius)
    J[:, 3] = -1

    ### End of your code ###
    return J
    
def fit_sphere(points):
    # Initial guess for the sphere parameters: [x0, y0, z0, r]
    x = np.array([0.0, 0.0, 0.0, 0.3])  # Start with center at origin, small radius

    for i in range(1000):
        ### Your code here ###
        # Compute the residuals and Jacobian matrix
        residual = sphere_residuals(x, points)
        J = sphere_jac(x, points)

        # Compute the update step with Gauss-Newton method
        delta = np.linalg.lstsq(J, residual, rcond=None)[0]

        # Update the parameters
        x = x - delta  

        ### End of your code ###
        
        # Check for convergence
        if np.linalg.norm(delta) < 1e-4:
            print(f"Converged after {i+1} iterations")
            break

    return x

def cylinder_residuals(params, points):
        x0, y0, z0 = params[:3]
        dx, dy, dz = params[3:6]
        r = params[6]
        axis = np.array([dx, dy, dz])

        ### Your code here ###
        # Normalise the axis vector
        axis_normalized = axis / np.linalg.norm(axis)

        ### Comment on line below: What does h represent and why it is calculated in this way?###
        # Answer: h is range of the point cloud along the cylinder's axis. h is calculated this 
        # way to measure how far the points are spread along the axis to ensure that h is large 
        # enough to fully encompass all the inlier points along its axis.
        h = np.ptp(np.dot(points - np.array([x0, y0, z0]), axis_normalized))
        
        # Compute the projection of each point onto the cylinder axis
        w = points - np.array([x0, y0, z0])
        
        # Distance along the axis
        projection_length = np.dot(w, axis_normalized)
        projection = np.outer(projection_length, axis_normalized)
        
        # Compute the distance of each point to the cylinder axis
        dist_to_axis = np.linalg.norm(w - projection, axis=1)
        
        # Compute the distance of each point to the cylinder bottom/top surfaces
        # Hint: if dist_to_height is used as a residual, shall we penalise the points that are right on the cylinder surface but with a postiive dist_to_height measures?
        dist_to_height = np.maximum(0.0, np.abs(projection_length) - 0.5*h)
        
        # Compute the residuals: distance of each point to the cylinder surface
        # Take the absolute value of the point to axis or height RESIDUE, whichever is greater
        dist_to_cyl = np.maximum(np.abs(dist_to_axis - r), dist_to_height)
        ### End of your code ###
        
        return dist_to_cyl
    
def fit_cylinder(points):
    initial_guess = [0, 0, 0, 0, 0, 1, 0.05]
    ### Bonus: Implement the Gauss-Newton optimization function to fit the cylinder parameters ###
    result = least_squares(cylinder_residuals, initial_guess, args=(points,))
    return result.x  # Estimated cylinder parameters

def ransac_lollipop_fitting(points, max_iterations=100, threshold=0.01):
    best_sphere_params = None
    best_cylinder_params = None
    best_inliers_count = 0
    best_cylinder_height = 0

    sphere_points = points
    
    for _ in range(max_iterations):
        ### Your code here ###
        # Sample 10 points from the sphere points
        sphere_indices = np.random.choice(len(sphere_points), 10, replace=False)
        sphere_sample = sphere_points[sphere_indices]
        
        # Fit a sphere to the sampled points
        sphere_params = fit_sphere(sphere_sample)
        
        # Compute the residuals of the sphere fit from all points
        sphere_distances = np.abs(sphere_residuals(sphere_params, points))
        
        # Identify inliers (points with distance to sphere within the threshold) based on the residuals
        sphere_inlier_mask = sphere_distances < threshold
        sphere_inliers = points[sphere_inlier_mask]
        
        # If the number of inliers is less than 50, continue to the next iteration
        if len(sphere_inliers) < 50:
            continue
        
        # Select the points that are not inliers to the sphere and use them to fit the cylinder
        cylinder_points = points[~sphere_inlier_mask]
        
        # Sample 10 points from the cylinder points
        cyl_indices = np.random.choice(len(cylinder_points), 10, replace=False)
        cylinder_sample = cylinder_points[cyl_indices]
        
        # Fit a cylinder to the sampled points
        cylinder_params = fit_cylinder(cylinder_sample)
        
        # Compute the residuals of the cylinder fit from all points
        dist_to_cyl = np.abs(cylinder_residuals(cylinder_params, points))
        
        # Compute the total number of inliers for both sphere and cylinder fits
        sphere_inliers = sphere_distances < threshold
        cylinder_inliers = dist_to_cyl < threshold

        total_inliers_count = np.sum(sphere_inliers) + np.sum(cylinder_inliers)
        
        # Update the best parameters if the current inliers count is higher
        if total_inliers_count > best_inliers_count:
            best_sphere_params = sphere_params.copy()
            best_cylinder_params = cylinder_params.copy()
            best_inliers_count = total_inliers_count

            # Calculate the cylinder height
            x0, y0, z0 = cylinder_params[:3]
            dx, dy, dz = cylinder_params[3:6]
            axis = np.array([dx, dy, dz])
            axis_normalized = axis / np.linalg.norm(axis)

            cylinder_inlier_points = points[cylinder_inliers]
            best_cylinder_height = np.ptp(np.dot(cylinder_inlier_points - np.array([x0, y0, z0]), axis_normalized))

        # Update the sphere points to exclude the inliers to the cylinder
        sphere_points = points[~cylinder_inliers]
        ### End of your code ###
        
    return best_sphere_params, best_cylinder_params, best_cylinder_height

def main():
    # Load the lollipop point cloud
    lollipop_pc = np.load("data/lollipop_data.npz")["points"]

    # Apply RANSAC to fit sphere and cylinder
    best_sphere_params, best_cylinder_params, cylinder_height = ransac_lollipop_fitting(
        lollipop_pc
    )

    # Extract the parameters
    sphere_center, sphere_radius = best_sphere_params[:3], best_sphere_params[3]
    cylinder_center, cylinder_axis, cylinder_radius = (
        best_cylinder_params[:3],
        best_cylinder_params[3:6],
        best_cylinder_params[6],
    )

    # Create Point Cloud object for visualization
    sphere_mesh = create_sphere_mesh(sphere_center, sphere_radius)
    cylinder_mesh = create_cylinder_mesh(
        cylinder_center, cylinder_axis, cylinder_radius, height=cylinder_height
    )

    # Convert point cloud to trimesh for visualization
    lollipop_cloud = trimesh.points.PointCloud(
        vertices=lollipop_pc, colors=[0, 0, 255, 255]
    )

    # scene = trimesh.Scene([lollipop_cloud])
    # scene.show()
    
    # Visualize using trimesh's scene
    scene = trimesh.Scene([lollipop_cloud, sphere_mesh, cylinder_mesh])
    scene.show()

if __name__ == "__main__":
    main()
