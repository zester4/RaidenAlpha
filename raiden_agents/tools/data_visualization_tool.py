# Code for raiden_agents/tools/data_visualization_tool.py
import logging
import json
import traceback
# Assuming matplotlib and seaborn are installed in the environment
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from raiden_agents.tools.base_tool import Tool, ToolExecutionError # Import base Tool and exceptions
from datetime import datetime # Needed for vector_db logging

logger = logging.getLogger("gemini_agent") # Assuming logger is configured elsewhere

class DataVisualizationTool(Tool):
    def __init__(self):
        super().__init__(
            name="visualize_data",
            description="Creates data visualizations using various plotting libraries",
            parameters={
                "type": "object",
                "properties": {
                    "data": {
                        "type": "string",
                        "description": "JSON string containing data to visualize. For arrays/lists, format as: [[x1,y1], [x2,y2], ...] or for dicts as: {\"col1\": [...], \"col2\": [...]}"
                    },
                    "plot_type": {
                        "type": "string",
                        "enum": ["line", "scatter", "bar", "histogram", "heatmap", "box"],
                        "description": "Type of visualization to create"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title for the plot"
                    },
                    "x_label": {
                        "type": "string",
                        "description": "Label for x-axis"
                    },\n                    "y_label": {
                        "type": "string",
                        "description": "Label for y-axis"
                    },
                    "output_file": {
                        "type": "string",
                        "description": "Output file path (png/jpg/pdf)"
                    }
                },\n                "required": ["data", "plot_type", "output_file"]
            }
        )

    def execute(self, **kwargs):
        self.validate_args(kwargs)
        # json, pandas, matplotlib.pyplot, seaborn imported globally or within the tool's environment

        try:
            # Parse data
            data_str = kwargs.get("data")
            if not data_str:
                 raise ToolExecutionError("No data provided.")

            data = json.loads(data_str)

            df = None
            # Handle list of lists [[x1,y1], ...] or dict {'col1': [...], ...}
            if isinstance(data, list) and all(isinstance(i, list) for i in data) and all(len(i) == 2 for i in data if i):
                 # Assuming list of [x,y] pairs
                 df = pd.DataFrame(data, columns=['x', 'y'])
            elif isinstance(data, dict):
                 # Assuming dict of lists for columns
                 df = pd.DataFrame(data)
            else:
                raise ToolExecutionError("Invalid data format. Expected list of [x,y] pairs or dictionary of lists.")

            # Check if DataFrame is empty
            if df.empty:
                 return "Provided data is empty. Cannot create visualization."

            # Create plot
            plt.figure(figsize=(10, 6))
            plot_type = kwargs.get("plot_type")
            title = kwargs.get("title", "")
            x_label = kwargs.get("x_label", "")
            y_label = kwargs.get("y_label", "")
            output_file = kwargs.get("output_file")

            # Ensure output file has a valid extension if not provided
            if not output_file.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf')):
                 output_file += '.png' # Default to png


            if plot_type == "line":
                if 'x' not in df.columns or 'y' not in df.columns:
                     raise ToolExecutionError("Line plot requires 'x' and 'y' columns in data.")
                plt.plot(df['x'], df['y'])
                if not x_label: x_label = 'X-axis'
                if not y_label: y_label = 'Y-axis'
            elif plot_type == "scatter":
                if 'x' not in df.columns or 'y' not in df.columns:
                     raise ToolExecutionError("Scatter plot requires 'x' and 'y' columns in data.")
                plt.scatter(df['x'], df['y'])
                if not x_label: x_label = 'X-axis'
                if not y_label: y_label = 'Y-axis'
            elif plot_type == "bar":
                 # Bar plot can work with 'x' (categories) and 'y' (values)
                 if 'x' not in df.columns or 'y' not in df.columns:
                     raise ToolExecutionError("Bar plot requires 'x' and 'y' columns in data.")
                 plt.bar(df['x'], df['y'])
                 if not x_label: x_label = 'Category'
                 if not y_label: y_label = 'Value'
            elif plot_type == "histogram":
                 # Histogram typically needs only one column (values)
                 if len(df.columns) == 1:
                     data_column = df.iloc[:, 0] # Use the first column
                 elif 'value' in df.columns:
                      data_column = df['value']
                 elif 'y' in df.columns:
                      data_column = df['y'] # Also support 'y' if from [x,y]
                 else:
                     raise ToolExecutionError("Histogram requires a single column of values.")
                 plt.hist(data_column, bins=30)
                 if not x_label: x_label = 'Value'
                 if not y_label: y_label = 'Frequency'
            elif plot_type == "heatmap":
                # Heatmap typically needs 'x', 'y', and 'value' columns
                if 'x' not in df.columns or 'y' not in df.columns or 'value' not in df.columns:
                     raise ToolExecutionError("Heatmap requires 'x', 'y', and 'value' columns.")
                try:
                     # Pivot the data to a matrix format suitable for heatmap
                     pivot_table = df.pivot(index='y', columns='x', values='value')
                     sns.heatmap(pivot_table, annot=True, fmt=".1f") # Add annotations with formatting
                 except Exception as pivot_e:
                      raise ToolExecutionError(f"Error creating heatmap pivot table: {pivot_e}. Ensure data is suitable for pivoting (unique x,y pairs).")

                if not x_label: x_label = 'X-Category'
                if not y_label: y_label = 'Y-Category'

            elif plot_type == "box":
                # Box plot can use 'x' (categories) and 'y' (values) or just 'y'
                if 'y' not in df.columns:
                     raise ToolExecutionError("Box plot requires at least a 'y' column.")

                if 'x' in df.columns:
                     sns.boxplot(x='x', y='y', data=df)
                     if not x_label: x_label = 'Category'
                     if not y_label: y_label = 'Value'
                else:
                     sns.boxplot(y=df['y'])
                     if not y_label: y_label = 'Value'
                     x_label = '' # No x-label if only one category

            else:
                # This case should be caught by validate_args enum, but for safety:
                raise ToolExecutionError(f"Unsupported plot type: {plot_type}")


            # Customize plot
            if title: plt.title(title)
            if x_label: plt.xlabel(x_label)
            if y_label: plt.ylabel(y_label)
            plt.tight_layout() # Adjust layout to prevent labels overlapping

            # Save plot
            plt.savefig(output_file)
            plt.close() # Close the plot figure to free memory

            logger.info(f"Plot saved to: {output_file}")

            # Assuming vector_db is available for logging/memory
            vector_db = None # Placeholder
            try:
                from __main__ import vector_db as main_vector_db
                vector_db = main_vector_db
            except ImportError:
                logger.warning("Could not import vector_db from __main__. Vector DB features in DataVisualizationTool will be disabled.")

            if vector_db and vector_db.is_ready():
                vector_db.add(
                    f"Generated data visualization: {plot_type} plot saved to {output_file}",
                    {
                        "type": "visualization",
                        "plot_type": plot_type,
                        "output_file": output_file,
                        "title": title,
                        "time": datetime.now().isoformat()
                    }
                )


            return f"Visualization created and saved to {output_file}"

        except json.JSONDecodeError as e:
            logger.error(f"Data parsing error: Invalid JSON format: {e}")
            raise ToolExecutionError(f"Invalid JSON data format: {e}. Please provide valid JSON for the 'data' parameter.")
        except KeyError as e:
             # Handle missing columns for plot types that require specific ones
             logger.error(f"Data column error for plot type {plot_type}: Missing column {e}")
             raise ToolExecutionError(f"Data error: Required column {e} is missing for {plot_type} plot. Please check your data format.")
        except ToolExecutionError as e:
             # Re-raise our own validation/execution errors
             raise e
        except Exception as e:
            logger.error(f"Visualization error: {e}", exc_info=True)
            traceback.print_exc() # Print traceback to logs
            raise ToolExecutionError(f"Failed to create visualization: {e}. Ensure data format is correct and libraries are installed.")
