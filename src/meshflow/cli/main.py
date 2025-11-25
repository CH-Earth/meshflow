import click
from pathlib import Path
from ..core import MESHWorkflow


@click.command()
@click.option('--json', 'json_file', required=True, type=click.Path(exists=True, path_type=Path),
              help='Path to the JSON configuration file')
@click.option('--output-path', default=None, type=click.Path(path_type=Path),
              help='Output path for saving results (optional)')
def main(json_file, output_path):
    """Run MESHWorkflow from a JSON configuration file."""
    # Load workflow from JSON file
    click.echo(f"Loading workflow from {json_file}")
    workflow = MESHWorkflow.from_json_file(json_file)

    if workflow.settings['core']['forcing_files'] == 'multiple':
        # Execute the workflow
        click.echo("Running workflow...")
        workflow.run(save_path=output_path)
    else:
        # Execute the workflow
        click.echo("Running workflow...")
        workflow.run()

    # Save results if output path is provided
    if output_path:
        click.echo(f"Saving results to {output_path}")
        workflow.save(output_path)
    else:
        click.echo("No output path specified, results not saved")

    click.echo("Workflow completed successfully!")

if __name__ == '__main__':
    main()
