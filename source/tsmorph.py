import numpy as np
import matplotlib.pyplot as plt

class TSmorph:
    def __init__(self, S: np.array, T: np.array, granularity: int) -> None:
        """
        Args:
            S (np.array): Source time series (shape: [dimensions, time_points])
            T (np.array): Target time series (shape: [dimensions, time_points])
            granularity (int): Number of semi-synthetic time series in the morphing process
        """
        self.S = S  # Shape: [dimensions, time_points]
        self.T = T  # Shape: [dimensions, time_points]
        self.granularity = granularity+2  # Add 2 to include the source and target series

    def transform(self) -> np.array:
        """
        Perform linear morphing for multivariate time series.

        Returns:
            np.array: Morphed time series (shape: [n, d, t])
                      where n = granularity, d = dimensions, t = time_points
        """
        # Morphing process (vectorized)
        alpha = np.linspace(0, 1, self.granularity).reshape(-1, 1, 1)  # Shape: [granularity, 1, 1]
        morphed_series = alpha * self.T + (1 - alpha) * self.S  # Shape: [granularity, dimensions, time_points]

        return morphed_series

    def plot_morphed_series(self, morphed_series: np.array, 
                            start_color='#61E6AA', end_color='#5722B1', 
                            title=True, morph_labels=True) -> None:
        """
        Plot the morphed time series with separate subplots for each dimension and gradient colors.

        Args:
            morphed_series (np.array): Morphed time series (shape: [n, d, t])
            start_color (str): Hex color code for the start of the gradient.
            end_color (str): Hex color code for the end of the gradient.
        """
        # Convert hex to RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4))

        # Generate gradient colors
        start_rgb = hex_to_rgb(start_color)
        end_rgb = hex_to_rgb(end_color)
        n_series = self.granularity

        colors = []
        for i in range(n_series):
            ratio = i / (n_series - 1)
            r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio
            g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio
            b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio
            colors.append((r, g, b))

        # Create subplots for each dimension
        dimensions = self.S.shape[0]
        fig, axes = plt.subplots(dimensions, 1, figsize=(12, 4 * dimensions), squeeze=False)

        for dim in range(dimensions):
            ax = axes[dim, 0]

            # Plot the original source and target series
            ax.plot(self.S[dim, :], color=start_color, linewidth=3, label='Source Series (S)')

            # Plot the morphed series with gradient colors
            for i in range(1, self.granularity - 1):
                source_pct = round((self.granularity - i - 1) / (self.granularity - 1) * 100)
                target_pct = round(i / (self.granularity - 1) * 100)
                
                if morph_labels:
                    ax.plot(morphed_series[i, dim, :], color=colors[i], linewidth=2,
                            label=f"S {source_pct}/{target_pct} T")
                else:
                    ax.plot(morphed_series[i, dim, :], color=colors[i], linewidth=2)

            # Plot the target series
            ax.plot(self.T[dim, :], color=end_color, linewidth=3, label='Target Series (T)')

            if title:
                ax.set_title(f'Dimension {dim}', fontsize=16, pad=15)
            ax.set_xlabel('Time', fontsize=16)
            ax.set_ylabel('Value', fontsize=16)
            ax.grid(True, alpha=0.3)

            # Move legend outside plot
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)

        plt.tight_layout()
        plt.show()

        # save plot to file 
        fig.savefig('morphed_series_plot.png', bbox_inches='tight')
        plt.close(fig)
        return