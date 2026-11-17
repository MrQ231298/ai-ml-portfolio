#!/usr/bin/env bash
# Single-node Kafka 4.x in KRaft mode (no ZooKeeper, no Docker). Needs Java 17+.
# Usage: KAFKA_HOME=/path/to/kafka_2.13-4.4.0 scripts/start_kafka.sh
set -euo pipefail
: "${KAFKA_HOME:?set KAFKA_HOME to an unpacked Apache Kafka 4.x release}"
DATA_DIR="${KAFKA_DATA:-/tmp/kafka-data}"
export KAFKA_HEAP_OPTS="${KAFKA_HEAP_OPTS:--Xms768m -Xmx768m}"
CFG="$KAFKA_HOME/config/server.properties"
if [ ! -f "$DATA_DIR/meta.properties" ]; then
  mkdir -p "$DATA_DIR"
  "$KAFKA_HOME/bin/kafka-storage.sh" format --standalone -t "$("$KAFKA_HOME/bin/kafka-storage.sh" random-uuid)" \
    -c "$CFG" --override "log.dirs=$DATA_DIR"
fi
exec "$KAFKA_HOME/bin/kafka-server-start.sh" "$CFG" --override "log.dirs=$DATA_DIR"
