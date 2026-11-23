"""
Configuration Module
====================
Purpose:
    This module defines the centralized configuration for the entire ML application.
    Using a configuration class makes it easy to change settings in one place.

Key Concepts:
    - @dataclass: A Python decorator that automatically generates __init__, __repr__, etc.
    - frozen=True: Makes the configuration immutable (can't be changed after creation)
    - Type hints: Specify what type each variable should be (int, float, Path, bool)

Why This Matters:
    Instead of hardcoding values throughout your code, centralizing configuration
    makes your code more maintainable and easier to test.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    """
    Centralized configuration for the entire ML application.
    
    Attributes:
        random_state (int): Seed for random number generators (ensures reproducibility)
                           Same seed = same random results each time
        output_dir (Path): Directory where outputs and artifacts are saved
        save_plots (bool): Whether to save plots to files
        show_plots (bool): Whether to display plots interactively
    
    Example:
        >>> config = AppConfig(random_state=42, save_plots=True)
        >>> print(config.random_state)
        42
    """
    
    # Default values for all configuration parameters
    random_state: int = 42
    output_dir: Path = Path("outputs")
    save_plots: bool = False
    show_plots: bool = True

    def __post_init__(self) -> None:
        """
        Validation method called automatically after __init__.
        
        This ensures that configuration values make sense before the app runs.
        It's better to catch errors early than to have mysterious failures later.
        
        Raises:
            ValueError: If any configuration value is invalid
        """
        # Random state should be non-negative (convention)
        if self.random_state < 0:
            raise ValueError(
                f"random_state must be non-negative, got {self.random_state}"
            )


# Example usage (for testing this module independently)
if __name__ == "__main__":
    # Create a config with default values
    config = AppConfig()
    print("Default Configuration:")
    print(f"  Random State: {config.random_state}")
    print(f"  Output Directory: {config.output_dir}")
    
    # Create a custom config
    custom_config = AppConfig(
        random_state=123,
        save_plots=True
    )
    print("\nCustom Configuration:")
    print(f"  Random State: {custom_config.random_state}")
    print(f"  Save Plots: {custom_config.save_plots}")
    
    # Try to create an invalid config (this will raise an error)
    try:
        invalid_config = AppConfig(random_state=-1)  # Invalid: negative
    except ValueError as e:
        print(f"\nCaught expected error: {e}") 