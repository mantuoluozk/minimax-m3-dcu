#!/usr/bin/env bash
set -euo pipefail

# Derived from the server's launch_docker.sh. Run on the host.
docker_image="${DOCKER_IMAGE:-minimax-m3-dcu:custom}"
container_name="${CONTAINER_NAME:-minimax-m3-dcu}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
work_dir=/workspace/minimax-m3-dcu
shm_size="${SHM_SIZE:-256G}"
log_dir="$repo_dir/logs"

mounts=(
    "/opt/hyhal:/opt/hyhal"
    "/data/xsf:/data/xsf"
    "${MODEL_HOST_DIR:-/model}:/model:ro"
    "$repo_dir:$work_dir:ro"
    "$log_dir:/var/log/minimax-m3"
)
devices=(/dev/kfd /dev/mkfd /dev/dri)
extra_args=(
    --ipc=host --group-add video --cap-add=SYS_PTRACE
    --security-opt seccomp=unconfined --network=host --privileged
)

if (( $# > 1 )); then
    echo 'Usage: bash docker.sh [print|-v|destroy]' >&2
    exit 2
fi
case "${1:-}" in
    print|p) printf '%s\n' "$container_name"; exit 0 ;;
    destroy|d) exec docker container rm -f "$container_name" ;;
    -v|'') ;;
    *) echo 'Usage: bash docker.sh [print|-v|destroy]' >&2; exit 2 ;;
esac

if docker container inspect "$container_name" >/dev/null 2>&1; then
    actual_image="$(docker container inspect --format '{{.Image}}' "$container_name")"
    expected_image="$(docker image inspect --format '{{.Id}}' "$docker_image")"
    if [[ "$actual_image" != "$expected_image" ]]; then
        printf 'Container %s uses a different image. Choose a new CONTAINER_NAME.\n' "$container_name" >&2
        exit 1
    fi
    mounted_repo="$(docker container inspect --format '{{range .Mounts}}{{if eq .Destination "/workspace/minimax-m3-dcu"}}{{.Source}}{{end}}{{end}}' "$container_name")"
    if [[ "$mounted_repo" != "$repo_dir" ]]; then
        printf 'Container %s mounts a different checkout. Choose a new CONTAINER_NAME.\n' "$container_name" >&2
        exit 1
    fi
    if [[ "$(docker container inspect --format '{{.State.Running}}' "$container_name")" != true ]]; then
        docker start "$container_name"
    fi
else
    mkdir -p "$log_dir"
    run=(docker run -id --name "$container_name" --shm-size "$shm_size" "${extra_args[@]}")
    for dev in "${devices[@]}"; do run+=(--device "$dev"); done
    for mount in "${mounts[@]}"; do run+=(-v "$mount"); done
    if [[ -n "${DOCKER_ENV_FILE:-}" ]]; then run+=(--env-file "$DOCKER_ENV_FILE"); fi
    run+=(-e LOG_DIR=/var/log/minimax-m3)
    # Keep an interactive shell available even when the image declares its own ENTRYPOINT.
    "${run[@]}" --workdir "$work_dir" --entrypoint /bin/bash "$docker_image" -i
fi

if [[ "${1:-}" == -v ]]; then
    printf 'docker exec -it -w %q %q /bin/bash\n' "$work_dir" "$container_name"
    exit 0
fi
exec docker exec -it -w "$work_dir" "$container_name" /bin/bash
