"""
Geometry utilities for fire modeling.
Includes validation, coordinate conversion, alignment, and plotting functions.
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString, MultiLineString
from shapely.ops import polygonize, unary_union
from shapely import make_valid


# ============================================================================
# GEOMETRY VALIDATION AND CONVERSION
# ============================================================================

def calculate_max_area_geom(multigeom):
    """
    Extract the largest polygon from a MultiPolygon or GeometryCollection.
    
    Args:
        multigeom: MultiPolygon or GeometryCollection
        
    Returns:
        Polygon with the largest area
    """
    if isinstance(multigeom, (GeometryCollection, MultiPolygon)):
        max_area = 0
        max_area_idx = 0
        for ix, g in enumerate(multigeom.geoms):
            if g.area > max_area:
                max_area = g.area
                max_area_idx = ix
        return calculate_max_area_geom(multigeom.geoms[max_area_idx])
    
    return multigeom


def lines_to_polygon(geom):
    """
    Convert LineString/MultiLineString perimeter boundaries to a Polygon/MultiPolygon.
    
    Args:
        geom: Shapely geometry (LineString, MultiLineString, or already polygonal)
        
    Returns:
        Polygon or MultiPolygon, or None if polygonization fails
    """
    if geom is None or geom.is_empty:
        print("Geometry is None or empty.")
        return None

    gt = geom.geom_type

    # Already polygonal
    if gt in ("Polygon", "MultiPolygon"):
        return geom

    # Line boundaries -> polygonize
    if gt in ("LineString", "MultiLineString"):
        merged = unary_union(geom)
        polys = list(polygonize(merged))
        if not polys:
            return None
        if len(polys) == 1:
            return polys[0]
        return MultiPolygon(polys)

    # GeometryCollection: try to extract polygonal parts first, else polygonize line parts
    if gt == "GeometryCollection":
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
        if polys:
            return unary_union(polys)

        lines = [g for g in geom.geoms if g.geom_type in ("LineString", "MultiLineString")]
        if lines:
            merged = unary_union(lines)
            polys2 = list(polygonize(merged))
            if not polys2:
                return None
            return MultiPolygon(polys2) if len(polys2) > 1 else polys2[0]

    return None


def validate_geom(poly):
    """
    Validate and clean a geometry.
    Converts lines to polygons, makes geometry valid, and extracts largest polygon.
    
    Args:
        poly: Shapely geometry
        
    Returns:
        Valid Polygon, or None if validation fails
    """
    poly = lines_to_polygon(poly)
    if poly is None:
        return None

    poly = make_valid(poly)
    if isinstance(poly, (GeometryCollection, MultiPolygon)):
        poly = calculate_max_area_geom(poly)
    
    if not isinstance(poly, Polygon):
        print(f'Validated geometry is not a Polygon. Type: {type(poly)}')
        
    return poly


def multipolygon_to_polygon(geom):
    """
    Convert MultiPolygon to single Polygon by taking the largest area polygon.
    
    Args:
        geom: Polygon or MultiPolygon
        
    Returns:
        Single Polygon
    """
    if isinstance(geom, Polygon):
        return geom
    elif isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")


# ============================================================================
# STATE CONVERSION (Geometry <-> State Vector)
# ============================================================================

def geom_to_state(geom):
    """
    Convert a Shapely Polygon to a state vector (numpy array).
    State format: [x0, y0, x1, y1, ..., xn, yn]
    
    Args:
        geom: Shapely Polygon
        
    Returns:
        State vector as numpy array of shape (2*n_vertices, 1)
    """
    geom = multipolygon_to_polygon(geom)
    coords = np.array(geom.exterior.coords[:-1])  # Exclude closing point
    return coords.reshape(2 * len(coords), 1)


def state_to_geom(state):
    """
    Convert a state vector to a Shapely Polygon.
    
    Args:
        state: State vector of shape (2*n_vertices, 1) or (2*n_vertices,)
        
    Returns:
        Validated Shapely Polygon
    """
    state = np.asarray(state).flatten()
    return validate_geom(Polygon(zip(state[::2], state[1::2])))


def xy_to_state(x, y):
    """
    Convert separate x and y coordinate lists to a state vector.
    
    Args:
        x: List of x coordinates
        y: List of y coordinates
        
    Returns:
        State vector as numpy array of shape (2*len(x), 1)
    """
    ret = []
    for i in range(len(x)):
        ret.append(x[i])
        ret.append(y[i])
    return np.array(ret).reshape((2*len(x), 1))


# ============================================================================
# GEOMETRY ALIGNMENT AND INTERPOLATION
# ============================================================================

def make_ccw(geom):
    """
    Ensure polygon vertices are in counter-clockwise order.
    
    Args:
        geom: Shapely Polygon
        
    Returns:
        Polygon with CCW vertices
    """
    if not geom.exterior.is_ccw:
        return geom.reverse()
    return geom


def interpolate_perimeter(vertices, target_count):
    """
    Interpolate vertices along a perimeter to achieve a target vertex count.
    
    Args:
        vertices: List of (x, y) tuples
        target_count: Desired number of vertices
        
    Returns:
        List of interpolated (x, y) tuples
    """
    vertices = np.array(vertices)
    step_len = np.sqrt(np.sum(np.diff(vertices, 1, 0)**2, 1))
    step_len = np.append([0], step_len)
    cumulative_len = np.cumsum(step_len)
    interpolation_loc = np.linspace(0, cumulative_len[-1], target_count)
    X = np.interp(interpolation_loc, cumulative_len, vertices[:, 0])
    Y = np.interp(interpolation_loc, cumulative_len, vertices[:, 1])
    return list(zip(X, Y))


def align_vertices(interpolated_vertices):
    """
    Align vertices across multiple geometries by minimizing L2 distance.
    Finds the optimal rotation for each geometry to align with the first.
    
    Args:
        interpolated_vertices: List of numpy arrays of vertices
        
    Returns:
        List of aligned vertex arrays
    """
    minroll_lst = []
    aligned_vertices = [interpolated_vertices[0]]
    
    for i in range(len(interpolated_vertices) - 1):
        right_vertices = interpolated_vertices[i + 1]
        
        # Find optimal roll by minimizing L2 distance
        l2perroll = []
        for roll in range(len(interpolated_vertices[i]) - 1):
            diff = aligned_vertices[0] - right_vertices
            diff2sum = (diff[:, 0]**2 + diff[:, 1]**2).sum()
            l2perroll.append(diff2sum)
            right_vertices = np.roll(right_vertices, 1, axis=0)
        
        minroll_lst.append(np.argmin(l2perroll))
    
    # Apply optimal rolls
    for i in range(len(interpolated_vertices) - 1):
        aligned_vertices.append(np.roll(interpolated_vertices[i + 1], minroll_lst[i], axis=0))
    
    return aligned_vertices


def interpolate_geom(geom, vertex_count):
    """
    Interpolate a geometry to have a specific number of vertices.
    
    Args:
        geom: Shapely Polygon
        vertex_count: Target number of vertices
        
    Returns:
        Interpolated Polygon
    """
    interpolated_geom = Polygon(interpolate_perimeter(geom.exterior.coords, vertex_count))
    
    # Handle off-by-one errors
    actual_count = len(interpolated_geom.exterior.coords[:-1])
    if actual_count == vertex_count - 1:
        interpolated_geom = Polygon(interpolate_perimeter(geom.exterior.coords, vertex_count + 1))
    elif actual_count == vertex_count + 1:
        interpolated_geom = Polygon(interpolate_perimeter(geom.exterior.coords, vertex_count - 1))
    
    return interpolated_geom


def interpolate_geoms(geoms, vertex_count):
    """
    Interpolate multiple geometries to have the same number of vertices.
    
    Args:
        geoms: List of Shapely Polygons
        vertex_count: Target number of vertices
        
    Returns:
        List of interpolated Polygons
    """
    return [interpolate_geom(geom, vertex_count) for geom in geoms]


def align_geoms(geoms, vertex_count):
    """
    Align all geometries based on the first geometry.
    Ensures all have the same vertex count and optimal rotation.
    
    Args:
        geoms: List of Shapely Polygons
        vertex_count: Target number of vertices
        
    Returns:
        List of aligned Polygons
    """
    # Interpolate to same vertex count
    interpolated_geoms = interpolate_geoms(geoms, vertex_count)
    
    # Extract vertices as CCW arrays
    interpolated_vertices = [make_ccw(interpolated_geoms[0]).exterior.coords[:-1]]
    for geom in interpolated_geoms[1:]:
        interpolated_vertices.append(make_ccw(geom).exterior.coords[:-1])
    
    # Align and return as Polygons
    return [Polygon(vertices) for vertices in align_vertices(np.array(interpolated_vertices))]


def align_states(state_lst, vertex_count=None):
    """
    Align two state vectors to have the same vertex count and optimal rotation.
    
    Args:
        state_lst: List of two state vectors
        vertex_count: Target vertex count (None = use maximum)
        
    Returns:
        List of two aligned state vectors
    """
    if vertex_count is None:
        vertex_count = max(len(st) for st in state_lst) // 2
    
    # Convert states to geometries
    x0, y0 = state_lst[0][::2], state_lst[0][1::2]
    x1, y1 = state_lst[1][::2], state_lst[1][1::2]
    
    geom0 = make_ccw(Polygon(zip(x0, y0)))
    geom1 = make_ccw(Polygon(zip(x1, y1)))
    
    # Align geometries
    geom0, geom1 = align_geoms([geom0, geom1], vertex_count)
    
    # Convert back to states
    x, y = geom0.exterior.coords.xy
    state0 = xy_to_state(x.tolist()[:-1], y.tolist()[:-1])
    
    x, y = geom1.exterior.coords.xy
    state1 = xy_to_state(x.tolist()[:-1], y.tolist()[:-1])
    
    return [state0, state1]


def resample_state_to_vertex_count(state, vertex_count):
    """
    Resample a state vector to have a specific vertex count.
    
    Args:
        state: State vector of shape (2*M, 1)
        vertex_count: Target number of vertices
        
    Returns:
        Resampled state vector of shape (2*vertex_count, 1)
    """
    geom = state_to_geom(state)
    geom = interpolate_geom(geom, vertex_count)
    return geom_to_state(geom)


# ============================================================================
# GEOMETRY BOUNDS
# ============================================================================

def geom_bounds(g):
    """Get geometry bounds as (minx, miny, maxx, maxy)."""
    return g.bounds


def state_bounds(st):
    """Get state vector bounds as (minx, miny, maxx, maxy)."""
    g = state_to_geom(st)
    return geom_bounds(g)


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_geometry(geom, ax=None, **kwargs):
    """
    Plot a Shapely geometry.
    
    Args:
        geom: Shapely Polygon or MultiPolygon
        ax: Matplotlib axis (creates new if None)
        **kwargs: Additional plotting arguments
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    if isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            x, y = g.exterior.coords.xy
            ax.plot(x, y, **kwargs)
    else:
        x, y = geom.exterior.coords.xy
        ax.plot(x, y, **kwargs)
    
    ax.set_aspect('equal')
    return ax


def plot_matrix(X, ax=None, show_stdev=False, **kwargs):
    """
    Plot ensemble matrix (state vectors as columns).
    
    Args:
        X: Matrix of shape (2*n_vertices, n_samples)
        ax: Matplotlib axis (creates new if None)
        show_stdev: Whether to plot standard deviation circles
        **kwargs: Additional plotting arguments
    """
    vcounts = X.shape[0] // 2
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    color = kwargs.get('color', (1, 0, 0, 0.9))
    
    # Plot mean
    X_mean = np.mean(X, axis=1)
    ax.plot(X_mean[::2], X_mean[1::2], **kwargs)
    
    # Plot standard deviation circles
    if show_stdev:
        x0, y0 = X_mean[::2], X_mean[1::2]
        radstd = np.zeros_like(x0)
        
        for vix in range(vcounts):
            print(f'Calculating {vix}/{vcounts}..', end='\r', flush=True)
            x, y = X[2*vix, :], X[2*vix+1, :]
            radius = np.sqrt((x - x0[vix])**2 + (y - y0[vix])**2)
            radstd[vix] = np.std(radius)
        print()
        
        for vix in range(vcounts):
            print(f'Drawing {vix}/{vcounts}..', end='\r', flush=True)
            circle = plt.Circle((x0[vix], y0[vix]), radius=radstd[vix], 
                              fill=False, edgecolor=(0, 0, 0, 0.4), lw=0.3)
            ax.add_artist(circle)
    
    ax.set_aspect('equal')
    return ax


def plot_matrix_ensemble(X, ax=None, plot_alix=None, alpha=0.1, **kwargs):
    """
    Plot all ensemble members.
    
    Args:
        X: Matrix of shape (2*n_vertices, n_samples)
        ax: Matplotlib axis (creates new if None)
        plot_alix: Vertex index to highlight with scatter plot
        alpha: Transparency for ensemble members
        **kwargs: Additional plotting arguments
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    # Plot all ensemble members
    for vix in range(X.shape[1]):
        ax.plot(X[::2, vix], X[1::2, vix], **kwargs)
    
    # Highlight specific vertex if requested
    if plot_alix is not None:
        ax.scatter(X[2*plot_alix, :], X[2*plot_alix+1, :], 
                  alpha=alpha, color=kwargs.get('color'), edgecolors=(0, 0, 0, 0))
    
    ax.set_aspect('equal')
    return ax


def save_svg(geom, filepath):
    """
    Save geometry as SVG file.
    (Placeholder - requires minidom import and specific viewport settings)
    
    Args:
        geom: Shapely geometry
        filepath: Output SVG file path
    """
    # This is a simplified version - original implementation required xml.dom.minidom
    # and specific viewport coordinates
    raise NotImplementedError("SVG export requires xml.dom.minidom - see original putils for full implementation")