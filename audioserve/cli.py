"""CLI entry point for AudioServe."""

from __future__ import annotations

import logging

import click


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]))
def main(log_level: str) -> None:
    """AudioServe — Optimized inference server for audio models."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@main.command()
@click.option("--model", "-m", required=True, help="Model ID (e.g. openai/whisper-large-v3)")
@click.option("--host", default="0.0.0.0", help="Server host")
@click.option("--port", "-p", default=8000, type=int, help="Server port")
@click.option("--dtype", default="float16", type=click.Choice(["float16", "float32", "int8", "int8_float16"]))
@click.option("--max-batch-size", default=32, type=int, help="Maximum batch size")
@click.option("--diarization/--no-diarization", default=False, help="Enable speaker diarization")
@click.option("--hf-token", default=None, help="HuggingFace token (required for pyannote)")
def serve(
    model: str,
    host: str,
    port: int,
    dtype: str,
    max_batch_size: int,
    diarization: bool,
    hf_token: str | None,
) -> None:
    """Start the AudioServe inference server."""
    from audioserve.engine import AudioServeEngine

    engine = AudioServeEngine(
        model=model,
        dtype=dtype,
        max_batch_size=max_batch_size,
        diarization=diarization,
        hf_token=hf_token,
    )
    engine.start()
    engine.serve(host=host, port=port)


@main.command()
@click.argument("audio_path")
@click.option("--model", "-m", default="openai/whisper-large-v3", help="Model ID")
@click.option("--language", "-l", default=None, help="Language code")
@click.option("--dtype", default="float16")
@click.option("--diarization/--no-diarization", default=False)
@click.option("--hf-token", default=None)
def transcribe(
    audio_path: str,
    model: str,
    language: str | None,
    dtype: str,
    diarization: bool,
    hf_token: str | None,
) -> None:
    """Transcribe an audio file locally (no server)."""
    import asyncio
    from audioserve.engine import AudioServeEngine

    engine = AudioServeEngine(model=model, dtype=dtype, diarization=diarization, hf_token=hf_token)
    engine.start()

    async def _run():
        if diarization:
            result = await engine.transcribe_with_diarization(
                audio_path, language=language
            )
            click.echo(f"\nSpeakers: {', '.join(result.speakers)}")
        else:
            result = await engine.transcribe(audio_path, language=language)

        click.echo(f"\n{result.text}")
        click.echo(f"\n[Duration: {result.duration:.1f}s | Processing: {result.processing_time:.2f}s]")

    asyncio.run(_run())
    engine.stop()


if __name__ == "__main__":
    main()
