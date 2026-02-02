import sys
import subprocess
import time
import os
from pathlib import Path
from dotenv import load_dotenv

# Use __file__ to get reliable path regardless of cwd
PROJECT_DIR = Path(__file__).parent.parent.absolute()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Load .env.test from Tests directory (where this file is located)
env_path = Path(__file__).parent / '.env.test'
load_dotenv(dotenv_path=env_path)

class TestPhase:
    """Represents a testing phase with specific test files"""
    
    def __init__(self, name, description, test_paths, required=True):
        self.name = name
        self.description = description
        self.test_paths = test_paths
        self.required = required
        self.results = None
        self.duration = 0
        self.tests_run = 0
        self.failures = 0
        self.errors = 0
        self.passed = False
    
    def run(self, verbose=True):
        print(f"\n{'='*70}")
        print(f"PHASE: {self.name}")
        print(f"Description: {self.description}")
        print(f"{'='*70}\n")
        start_time = time.time()
        
        # Build pytest command using python -m pytest for reliable execution
        cmd = [sys.executable, '-m', 'pytest', '-q']
        cmd.extend(self.test_paths)
        cmd.extend([
            '--tb=line',  # One line per failure
            '-p', 'no:warnings',  # No warnings
            '-p', 'no:cov',  # No coverage
            '--maxfail=999',  # Continue through all failures
        ])

        try:
            # Create environment with settings to prevent hanging on Windows
            env = os.environ.copy()
            env['TQDM_DISABLE'] = '1'
            env['PYTHONUNBUFFERED'] = '1'  # Prevent buffering issues

            result = subprocess.run(
                cmd,
                cwd=PROJECT_DIR,
                stdin=subprocess.DEVNULL,  # Don't wait for input
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                timeout=600,
                env=env
            )
            
            self.duration = time.time() - start_time
            
            # Get output (stderr is merged with stdout)
            output = result.stdout or ''
            
            # Debug: Always show pytest output
            if output.strip():
                print("Pytest output:")
                print(output)
                print()
            
            # Parse pytest output
            self._parse_pytest_output(output)
            
            # Print minimal results
            if self.tests_run > 0:
                print(f"Tests Run: {self.tests_run}")
                print(f"Passed: {self.tests_run - self.failures - self.errors}")
                print(f"Failed: {self.failures}")
                print(f"Errors: {self.errors}")
                
                # Show failed test names if any
                if self.failures > 0 or self.errors > 0:
                    self._print_failures(output)
                
                # 75% pass rate required
                success_ratio = (self.tests_run - self.failures - self.errors) / self.tests_run
                self.passed = (success_ratio >= 0.75)
            else:
                print("No tests run")
                self.passed = False
            
            return self.passed
            
        except subprocess.TimeoutExpired:
            print(f"ERROR: Tests timed out after 600 seconds")
            self.duration = time.time() - start_time
            return False
        except Exception as e:
            print(f"ERROR: Failed to run tests: {e}")
            self.duration = time.time() - start_time
            return False
    
    def _print_failures(self, output):
        """Extract and print failed test names from pytest output"""
        import re
        
        # Look for FAILED lines in pytest output
        failed_pattern = r'FAILED (.*?) -'
        failures = re.findall(failed_pattern, output)
        
        if failures:
            print("\nFailed Tests:")
            for failure in failures:
                # Clean up the test name
                test_name = failure.replace('Tests\\', '').replace('/', '.')
                print(f"  - {test_name}")

    
    def _parse_pytest_output(self, output):
        """Parse pytest output to extract test counts"""
        # Look for lines like: "5 passed in 2.34s" or "3 passed, 2 failed in 1.23s"
        import re
        
        # Match patterns like "X passed", "X failed", "X error"
        passed_match = re.search(r'(\d+) passed', output)
        failed_match = re.search(r'(\d+) failed', output)
        error_match = re.search(r'(\d+) error', output)
        
        if passed_match:
            self.tests_run += int(passed_match.group(1))
        if failed_match:
            self.failures = int(failed_match.group(1))
            self.tests_run += self.failures
        if error_match:
            self.errors = int(error_match.group(1))
            self.tests_run += self.errors
    
    def print_summary(self):
        if self.tests_run == 0:
            print(f"\n{self.name}: NO TESTS RUN")
            return
        
        status = "PASSED" if self.passed else "FAILED"
        print(f"\n{self.name}: {status}")
        print(f"  Tests: {self.tests_run} | Failures: {self.failures} | Errors: {self.errors} | Time: {self.duration:.2f}s")


class TestOrchestrator:
    """Orchestrates all test phases for pre-pipeline validation"""
    
    def __init__(self):
        self.phases = self._define_phases()
        self.total_duration = 0
    
    def _define_phases(self):
        """Define test phases for pre-pipeline validation"""
        
        # Paths relative to Tests directory (where this script is)
        
        # Phase 1: Critical Unit Tests (Core Components)
        phase1 = TestPhase(
            name="Phase 1: Critical Unit Tests",
            description="Core component validation - activation functions, database basics",
            test_paths=[
                'Tests/unit/test_activation_func.py',
                'Tests/unit/test_database_creation.py',
            ],
            required=True
        )
        
        # Phase 2: ML Pipeline Unit Tests
        phase2 = TestPhase(
            name="Phase 2: ML Pipeline Unit Tests",
            description="Preprocessing, feature extraction, and neural network components",
            test_paths=[
                'Tests/unit/test_preprocessing_from_db.py',
                'Tests/unit/test_feature_extraction.py',
                'Tests/unit/test_network.py',
            ],
            required=True
        )
        
        # Phase 3: Backend API Unit Tests
        phase3 = TestPhase(
            name="Phase 3: Backend API Unit Tests",
            description="All API endpoints and backend services",
            test_paths=[
                'Tests/api/test_add_fruit.py',
                'Tests/api/test_admin_dashboard.py',
                'Tests/api/test_user_dashboard.py',
                'Tests/api/test_camera_api.py',
                'Tests/api/test_settings_api.py',
                'Tests/api/test_results_api.py',
                'Tests/api/test_processing_api.py',
            ],
            required=False  # Not required for pipeline, but good to have
        )
        
        # Phase 4: Integration Tests
        phase4 = TestPhase(
            name="Phase 4: Integration Tests",
            description="Database processing, feature models, and processing features",
            test_paths=[
                'Tests/integration/test_db_processing.py',
                'Tests/integration/test_feature_model.py',
                'Tests/integration/test_processing_features.py',
            ],
            required=True
        )
        
        # Phase 5: Full Pipeline Integration
        phase5 = TestPhase(
            name="Phase 5: Full Pipeline Integration",
            description="End-to-end pipeline validation before model building",
            test_paths=[
                'Tests/integration/test_build_model.py',
            ],
            required=False  # Optional
        )
        
        return [phase1, phase2, phase3, phase4, phase5]
    
    def run_phase(self, phase_number, verbose=True):
        """Run a specific phase"""
        if phase_number < 1 or phase_number > len(self.phases):
            print(f"Error: Phase {phase_number} does not exist")
            return False
        
        phase = self.phases[phase_number - 1]
        return phase.run(verbose=verbose)
    
    def run_quick_validation(self, verbose=True):
        """
        Run quick pre-pipeline validation - only unit and integration tests.
        Skips API tests and full pipeline integration.
        Perfect for validating before running build_model pipeline.
        """
        print("\n" + "="*70)
        print("QUICK PRE-PIPELINE VALIDATION")
        print("Unit Tests + Integration Tests Only")
        print("="*70)
        start_time = time.time()
        all_passed = True
        
        # Run Phase 1, 2, and 4 only (skip API tests and full pipeline)
        phases_to_run = [1, 2, 4]  # Critical Unit, ML Pipeline, Integration
        
        for phase_num in phases_to_run:
            phase = self.phases[phase_num - 1]
            success = phase.run(verbose=verbose)
            if not success:
                all_passed = False
                print(f"\nSTOPPING: Phase {phase_num} failed")
                break
        
        self.total_duration = time.time() - start_time
        self._print_summary(all_passed, mode="quick")
        
        return all_passed
    
    def run_critical_only(self, verbose=True):
        """Run only critical phases (skip optional API tests)"""
        print("\n" + "="*70)
        print("CRITICAL TESTS ONLY - Pre-Pipeline Validation")
        print("="*70)
        
        start_time = time.time()
        all_passed = True
        
        for i, phase in enumerate(self.phases, 1):
            if phase.required:
                success = phase.run(verbose=verbose)
                if not success:
                    all_passed = False
                    print(f"\nSTOPPING: Critical phase {i} failed")
                    break
            else:
                print(f"\nSkipping optional phase: {phase.name}")
        
        self.total_duration = time.time() - start_time
        self._print_summary(all_passed, mode="critical")
        
        return all_passed
    
    def run_all(self, verbose=True, stop_on_failure=True):
        """Run all test phases"""
        start_time = time.time()
        
        print("\n" + "="*70)
        print("COMPREHENSIVE PRE-PIPELINE TEST SUITE")
        print("="*70)
        
        all_passed = True
        
        for i, phase in enumerate(self.phases, 1):
            success = phase.run(verbose=verbose)
            
            if not success:
                all_passed = False
                if stop_on_failure and phase.required:
                    print(f"\nSTOPPING: Required phase {i} failed")
                    break
                elif not phase.required:
                    print(f"\nContinuing: Optional phase {i} failed")
        
        self.total_duration = time.time() - start_time
        self._print_summary(all_passed, mode="full")
        
        return all_passed
    
    def _print_summary(self, all_passed, mode="full"):
        """Print comprehensive test summary"""
        print("\n" + "="*70)
        print(f"TEST SUMMARY - {mode.upper()} MODE")
        print("="*70)
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        
        for phase in self.phases:
            if phase.tests_run > 0:  # Only show phases that ran
                phase.print_summary()
                total_tests += phase.tests_run
                total_failures += phase.failures
                total_errors += phase.errors
        
        print("\n" + "-"*70)
        print("OVERALL STATISTICS")
        print("-"*70)
        print(f"Total Tests Run: {total_tests}")
        print(f"Successes: {total_tests - total_failures - total_errors}")
        print(f"Failures: {total_failures}")
        print(f"Errors: {total_errors}")
        print(f"Total Time: {self.total_duration:.2f}s")
        
        if total_tests > 0:
            pass_rate = ((total_tests - total_failures - total_errors) / total_tests * 100)
            print(f"Pass Rate: {pass_rate:.1f}%")
        
        print("-"*70)
        
        if all_passed:
            print("\nSTATUS: ALL TESTS PASSED")
            print("System is validated and ready for pipeline execution")
        else:
            print("\nSTATUS: SOME TESTS FAILED")
            print("Fix issues before running pipeline")
        
        print("="*70 + "\n")
    
    def print_phase_info(self):
        """Print information about available phases"""
        print("\n" + "="*70)
        print("AVAILABLE TEST PHASES")
        print("="*70)
        
        for i, phase in enumerate(self.phases, 1):
            req_status = "REQUIRED" if phase.required else "OPTIONAL"
            print(f"\nPhase {i}: {phase.name} [{req_status}]")
            print(f"  Description: {phase.description}")
            print(f"  Test files ({len(phase.test_paths)}):")
            for path in phase.test_paths:
                print(f"    - {Path(path).name}")
        
        print("\n" + "="*70 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Pre-Pipeline Test Orchestrator - Validates system before pipeline execution'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help='Run specific phase (1=Critical Unit, 2=ML Pipeline, 3=API, 4=Integration, 5=Full Pipeline)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick validation before pipeline - Unit + Integration tests only (skips API tests)'
    )
    parser.add_argument(
        '--critical-only',
        action='store_true',
        help='Run only critical tests (skip optional API tests)'
    )
    parser.add_argument(
        '--continue-on-failure',
        action='store_true',
        help='Continue running phases even if one fails'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show information about test phases and exit'
    )
    
    args = parser.parse_args()
    
    orchestrator = TestOrchestrator()
    
    # Show phase info
    if args.info:
        orchestrator.print_phase_info()
        sys.exit(0)
    
    # Run specific phase
    if args.phase:
        print(f"\nRunning Phase {args.phase} only...")
        success = orchestrator.run_phase(args.phase, verbose=not args.quiet)
        sys.exit(0 if success else 1)
    
    # Run quick validation (unit + integration only)
    if args.quick:
        success = orchestrator.run_quick_validation(verbose=not args.quiet)
        sys.exit(0 if success else 1)
    
    # Run critical tests only
    if args.critical_only:
        success = orchestrator.run_critical_only(verbose=not args.quiet)
        sys.exit(0 if success else 1)
    
    # Run all tests
    success = orchestrator.run_all(
        verbose=not args.quiet,
        stop_on_failure=not args.continue_on_failure
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()