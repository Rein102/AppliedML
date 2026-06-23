import json
import socket
import sys

import click


@click.command()
@click.argument("image_path")
@click.option("--experiment-name", type=str, default="output/zero-shot")
@click.option("--ref-images", type=str, default=None)
@click.option("--port", type=int, default=5000)
def main(image_path, experiment_name, ref_images, port):
    ref_image_paths = [p.strip() for p in ref_images.split(",")] if ref_images else []

    request = json.dumps({
        "image": image_path,
        "experiment": experiment_name,
        "ref_images": ref_image_paths,
    }).encode()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
        sock.sendall(request)
        sock.shutdown(socket.SHUT_WR)

        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
        sock.close()

        result = json.loads(response.decode())
        if "error" in result:
            print(f"Error: {result['error']}", file=sys.stderr)
            sys.exit(1)

        print(f"\n=== Results ===")
        print(f"Classification score : {result['score']:.4f}  (0 = normal, 1 = anomaly)")
        print(f"Segmentation map     : {result['segmentation']}")
        print(f"Heatmap overlay      : {result['overlay']}")

    except ConnectionRefusedError:
        print("Error: Server not running. Start it first with:", file=sys.stderr)
        print("  poetry run python scripts/serve.py --classname carpet --pretrained laion400m_e31", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
