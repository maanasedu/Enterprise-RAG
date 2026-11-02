"""Run the Dimension 1 retrieval evaluation.

    python scripts/run_retrieval_evaluation.py --split calibration
    python scripts/run_retrieval_evaluation.py --split test --confirm-final-test

See docs/retrieval_evaluation.md for the protocol.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from evaluation.retrieval_eval import main

if __name__ == '__main__':
    sys.exit(main())
