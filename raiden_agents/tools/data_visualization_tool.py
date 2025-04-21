import logging
import json
import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from .base_tool import Tool, ToolExecutionError
from datetime import datetime

logger = logging.getLogger("gemini_agent")

# Attempt to import vector_db for logging
try:
    from __main__ import vector_db
except ImportError:
    vector_db = None

class DataVisualizationTool(Tool):
    def __init__(self):
        super().__init__(
            name="visualize_data",
            description="Creates data visualizations (line, scatter, bar, histogram, heatmap, box, pie, area, violin, pairplot) from JSON strings or local CSV/JSON files. Allows customization.",
            parameters={
                "type": "object",
                "properties": {
                    "data_source": {
                        "type": "string",
                        "description": "Required. Either a JSON string containing the data, or a local file path (e.g., 'data/my_data.csv', 'results.json')."
                    },
                    "data_format": {
                        "type": "string",
                        "enum": ["json_string", "csv_file", "json_file"],
                        "description": "Optional. Specify the format of 'data_source'. If omitted, it will be inferred (JSON string vs. file path based on content/extension)."
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["line", "scatter", "bar", "histogram", "heatmap", "box", "pie", "area", "violin", "pairplot"],
                        "description": "Required. Type of visualization to create."
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Required. Output file path for the plot (e.g., 'plots/my_chart.png'). Extension determines format (png, jpg, pdf)."
                    },
                    # --- Optional Column Specifications ---
                    "x_col": {"type": "string", "description": "Name of the column for the x-axis."},
                    "y_col": {"type": "string", "description": "Name of the column for the y-axis."},
                    "value_col": {"type": "string", "description": "Name of the column for values (e.g., for histogram, pie)."},
                    "category_col": {"type": "string", "description": "Name of the column for categories (e.g., for bar, box)."},
                    "hue_col": {"type": "string", "description": "Name of column for color grouping/hue."},
                    "size_col": {"type": "string", "description": "Name of column for marker size (scatter)."},
                    # --- Optional Customizations ---
                    "title": {"type": "string", "description": "Title for the plot."},
                    "x_label": {"type": "string", "description": "Custom label for x-axis."},
                    "y_label": {"type": "string", "description": "Custom label for y-axis."},
                    "figure_size": {"type": "string", "description": "Figure size in inches (e.g., '12,8')."},
                    "color": {"type": "string", "description": "Single color name or hex code."},
                    "palette": {"type": "string", "description": "Seaborn color palette name (e.g., 'viridis')."},
                    "legend": {"type": "boolean", "description": "Show legend (default: true where applicable)."},
                    "grid": {"type": "boolean", "description": "Show grid (default: true)."},
                    "style": {"type": "string", "description": "Seaborn style (e.g., 'whitegrid', 'darkgrid')."}
                },
                "required": ["data_source", "plot_type", "output_file"]
            }
        )

    def _load_data(self, data_source, data_format=None):
        """Loads data from JSON string or file path into a pandas DataFrame."""
        df = None
        source_type = "unknown"

        # Infer format if not provided
        if not data_format:
            if data_source.strip().startswith(('[', '{')):
                data_format = "json_string"
                source_type = "JSON string"
            elif Path(data_source).suffix.lower() == '.csv':
                data_format = "csv_file"
                source_type = f"CSV file ({data_source})"
            elif Path(data_source).suffix.lower() == '.json':
                data_format = "json_file"
                source_type = f"JSON file ({data_source})"
            else:
                 # Attempt JSON string as last resort before error
                 try:
                      json.loads(data_source)
                      data_format = "json_string"
                      source_type = "JSON string"
                 except json.JSONDecodeError:
                      raise ToolExecutionError(f"Could not infer data format for data_source. Please specify 'data_format' (json_string, csv_file, json_file) or provide valid JSON string/file path.")

        logger.info(f"Attempting to load data from: {source_type}")

        try:
            if data_format == "json_string":
                data = json.loads(data_source)
                df = pd.DataFrame(data)
            elif data_format == "csv_file":
                file_path = Path(data_source).resolve()
                if not file_path.is_file(): raise FileNotFoundError()
                df = pd.read_csv(file_path)
            elif data_format == "json_file":
                file_path = Path(data_source).resolve()
                if not file_path.is_file(): raise FileNotFoundError()
                df = pd.read_json(file_path) # Can handle various JSON orientations

            if df is None or df.empty:
                raise ValueError("Loaded data is empty or invalid.")

            logger.info(f"Successfully loaded data. Shape: {df.shape}, Columns: {list(df.columns)}")
            return df

        except json.JSONDecodeError as e:
            raise ToolExecutionError(f"Invalid JSON data provided in 'data_source'. Error: {e}")
        except FileNotFoundError:
            raise ToolExecutionError(f"Data file not found at path: {data_source}")
        except ValueError as e:
             raise ToolExecutionError(f"Error processing data: {e}")
        except Exception as e:
            logger.error(f"Error loading data from {source_type}: {e}", exc_info=True)
            raise ToolExecutionError(f"Failed to load or parse data from '{data_source}'. Error: {e}")

    def _get_columns(self, df, plot_type, x_col=None, y_col=None, value_col=None, category_col=None, hue_col=None, size_col=None):
        """Determines which columns to use based on plot type and explicit/inferred names."""
        cols = df.columns
        num_cols = len(cols)
        x, y, value, category, hue, size = x_col, y_col, value_col, category_col, hue_col, size_col

        # --- Inference Logic ---
        if plot_type in ["line", "scatter", "bar", "area"]:
            if not x and not y and num_cols == 2:
                x, y = cols[0], cols[1]
                logger.info(f"Inferred x='{x}', y='{y}' for {plot_type} plot from 2 columns.")
            elif not x and category and num_cols >= 2: # Allow using category_col as x
                 x = category
            elif not y and value and num_cols >= 2: # Allow using value_col as y
                 y = value

        elif plot_type in ["box", "violin"]:
             if not y and not value and num_cols == 1: # Plot single column
                  y = cols[0]
                  logger.info(f"Using single column '{y}' for {plot_type} plot.")
             elif not x and not category and y and num_cols >= 2: # Category specified by x/category_col
                  pass # y is already set or will be checked later
             elif x and not y and not value and num_cols >= 2: # y specified by y/value_col
                  pass
             elif not x and not category and not y and not value and num_cols == 2: # Infer category/value
                  x, y = cols[0], cols[1]
                  logger.info(f"Inferred category='{x}', value='{y}' for {plot_type} plot from 2 columns.")
             # Use explicit category/value if provided
             if not x and category: x = category
             if not y and value: y = value

        elif plot_type == "histogram":
             if not value and num_cols == 1:
                  value = cols[0]
                  logger.info(f"Using single column '{value}' for histogram.")
             # If multiple columns, 'value_col' must be specified

        elif plot_type == "pie":
             if not value and num_cols == 1: # Use index as labels, column as values
                  value = cols[0]
                  logger.info(f"Using index for labels and column '{value}' for pie chart values.")
             elif not value and not category and num_cols == 2: # Infer category/value
                  category, value = cols[0], cols[1]
                  logger.info(f"Inferred category='{category}', value='{value}' for pie chart.")
             # Use explicit category/value if provided
             if not category and x: category = x # Allow x_col for category
             if not value and y: value = y # Allow y_col for value

        # --- Validation ---
        required = {}
        if plot_type in ["line", "scatter", "bar", "area"]: required = {'x': x, 'y': y}
        elif plot_type in ["box", "violin"]: required = {'y': y} # x/category is optional
        elif plot_type == "histogram": required = {'value': value}
        elif plot_type == "heatmap": required = {'x': x, 'y': y, 'value': value}
        elif plot_type == "pie": required = {'value': value} # category/labels optional
        elif plot_type == "pairplot": pass # Uses all numerical columns by default, or specific vars (not implemented yet)

        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ToolExecutionError(f"Could not determine required column(s) '{', '.join(missing)}' for plot type '{plot_type}'. Please provide explicit column names (e.g., x_col, y_col, value_col).")

        # Validate optional columns exist if specified
        for col_name in [x, y, value, category, hue, size]:
            if col_name and col_name not in cols:
                raise ToolExecutionError(f"Specified column '{col_name}' not found in data.")

        return x, y, value, category, hue, size


    def execute(self, **kwargs):
        # Validate required args (data_source, plot_type, output_file)
        required_base = ["data_source", "plot_type", "output_file"]
        missing_base = [p for p in required_base if not kwargs.get(p)]
        if missing_base:
             raise ToolExecutionError(f"Missing required parameters: {', '.join(missing_base)}")

        data_source = kwargs.get("data_source")
        data_format = kwargs.get("data_format")
        plot_type = kwargs.get("plot_type")
        output_file = kwargs.get("output_file")

        # Optional args
        x_col, y_col, value_col, category_col, hue_col, size_col = (
            kwargs.get("x_col"), kwargs.get("y_col"), kwargs.get("value_col"),
            kwargs.get("category_col"), kwargs.get("hue_col"), kwargs.get("size_col")
        )
        title = kwargs.get("title", f"{plot_type.capitalize()} Plot") # Default title
        x_label, y_label = kwargs.get("x_label"), kwargs.get("y_label")
        fig_size_str = kwargs.get("figure_size")
        color = kwargs.get("color")
        palette = kwargs.get("palette")
        show_legend = kwargs.get("legend", True)
        show_grid = kwargs.get("grid", True)
        style = kwargs.get("style", "whitegrid") # Default seaborn style

        try:
            # --- Load Data ---
            df = self._load_data(data_source, data_format)

            # --- Determine Columns ---
            x, y, value, category, hue, size = self._get_columns(
                df, plot_type, x_col, y_col, value_col, category_col, hue_col, size_col
            )

            # --- Setup Plot Style ---
            sns.set_style(style)
            fig_size = None
            if fig_size_str:
                try:
                    fig_size = tuple(map(float, fig_size_str.split(',')))
                    if len(fig_size) != 2: raise ValueError()
                except:
                    logger.warning(f"Invalid figure_size format '{fig_size_str}'. Using default.")
                    fig_size = (10, 6) # Default size
            plt.figure(figsize=fig_size or (10, 6)) # Use specified or default

            # --- Generate Plot ---
            ax = None # Store axis for customization if needed
            plot_args = {'data': df, 'palette': palette, 'color': color}

            if plot_type == "line":
                ax = sns.lineplot(x=x, y=y, hue=hue, size=size, **plot_args)
                if not x_label: x_label = x
                if not y_label: y_label = y
            elif plot_type == "scatter":
                ax = sns.scatterplot(x=x, y=y, hue=hue, size=size, **plot_args)
                if not x_label: x_label = x
                if not y_label: y_label = y
            elif plot_type == "bar":
                # Use category for x if available, otherwise inferred x
                plot_args.pop('size', None) # Size not applicable
                ax = sns.barplot(x=x, y=y, hue=hue, **plot_args)
                if not x_label: x_label = x
                if not y_label: y_label = y
            elif plot_type == "histogram":
                plot_args.pop('palette', None); plot_args.pop('size', None) # Not applicable
                ax = sns.histplot(x=value, hue=hue, bins=30, kde=True, **plot_args)
                if not x_label: x_label = value
                if not y_label: y_label = 'Frequency'
            elif plot_type == "heatmap":
                plot_args.pop('color', None); plot_args.pop('size', None); plot_args.pop('hue', None) # Not applicable
                try:
                    pivot_table = df.pivot(index=y, columns=x, values=value)
                    ax = sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap=palette or "viridis", **plot_args)
                except Exception as pivot_e:
                    raise ToolExecutionError(f"Error creating heatmap pivot table: {pivot_e}. Ensure data has unique x/y pairs.")
                if not x_label: x_label = x
                if not y_label: y_label = y
            elif plot_type == "box":
                plot_args.pop('size', None) # Not applicable
                ax = sns.boxplot(x=x, y=y, hue=hue, **plot_args)
                if not x_label and x: x_label = x
                if not y_label: y_label = y
            elif plot_type == "pie":
                 plot_args.pop('color', None); plot_args.pop('size', None); plot_args.pop('hue', None) # Not applicable
                 # Use category for labels if available, otherwise index
                 labels = df[category].tolist() if category else df.index
                 plt.pie(df[value], labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette(palette) if palette else None)
                 plt.ylabel('') # Pie charts don't typically have a y-label
                 if not x_label and category: x_label = category # Use category as effective 'x' label
            elif plot_type == "area":
                 plot_args.pop('palette', None); plot_args.pop('size', None); plot_args.pop('hue', None) # Simplify
                 plt.fill_between(df[x], df[y], color=color or sns.color_palette()[0], alpha=0.4)
                 sns.lineplot(x=x, y=y, color=color or sns.color_palette()[0], **plot_args) # Overlay line
                 if not x_label: x_label = x
                 if not y_label: y_label = y
            elif plot_type == "violin":
                 plot_args.pop('size', None) # Not applicable
                 ax = sns.violinplot(x=x, y=y, hue=hue, **plot_args)
                 if not x_label and x: x_label = x
                 if not y_label: y_label = y
            elif plot_type == "pairplot":
                 plot_args.pop('color', None); plot_args.pop('size', None); # Use hue/palette
                 # Uses all numerical columns by default. Add 'vars' param later if needed.
                 sns.pairplot(hue=hue, palette=palette, data=df) # Pairplot creates its own figure/axes
                 # Title/labels less applicable here, set via pairplot directly if needed

            # --- Customize and Save ---
            if plot_type != "pairplot": # Pairplot handles its own figure/titles
                if title: plt.title(title)
                if x_label: plt.xlabel(x_label)
                if y_label: plt.ylabel(y_label)
                if ax and not show_legend: ax.legend().set_visible(False)
                if ax and show_grid: ax.grid(True)
                plt.tight_layout()

            # Ensure output path has valid extension
            output_path_obj = Path(output_file)
            valid_extensions = {'.png', '.jpg', '.jpeg', '.pdf', '.svg'}
            if output_path_obj.suffix.lower() not in valid_extensions:
                 logger.warning(f"Invalid output file extension '{output_path_obj.suffix}'. Defaulting to '.png'.")
                 output_path_obj = output_path_obj.with_suffix('.png')

            # Ensure output directory exists
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            plt.savefig(output_path_obj)
            plt.close() # Close the plot figure to free memory
            final_output_path = str(output_path_obj)
            logger.info(f"Plot saved to: {final_output_path}")

            # --- Log to Vector DB ---
            if vector_db and vector_db.is_ready():
                vector_db.add(
                    f"Generated data visualization: {plot_type} plot saved to {final_output_path}",
                    {
                        "type": "visualization",
                        "plot_type": plot_type,
                        "output_file": final_output_path,
                        "title": title,
                        "time": datetime.now().isoformat()
                    }
                )

            return f"Visualization '{title}' ({plot_type}) created and saved to {final_output_path}"

        except ToolExecutionError as e:
             raise e # Re-raise specific tool errors
        except Exception as e:
            logger.error(f"Visualization error: {e}", exc_info=True)
            traceback.print_exc()
            raise ToolExecutionError(f"Failed to create visualization: {e}. Check data format, column names, and parameters.")
