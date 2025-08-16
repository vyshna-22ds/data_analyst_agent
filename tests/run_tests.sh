#!/usr/bin/env bash
set -euo pipefail
API=${1:-http://127.0.0.1:7860/api/}
echo "Testing Highest-grossing films example..."
curl -s "$API" -F "questions.txt=@samples/questions/question.txt" | jq .
echo "Done."
