#!/usr/bin/env python3
"""
Frontend Test Suite - Master Runner
Automated test execution with server startup and reporting
"""

import subprocess
import sys
import time
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Test configuration"""
    
    # Directories
    PROJECT_ROOT = Path(__file__).parent
    BACKEND_DIR = PROJECT_ROOT.parent / "backend"
    FRONTEND_DIR = PROJECT_ROOT
    TESTS_DIR = PROJECT_ROOT / "Tests"
    
    # Server URLs
    BACKEND_URL = "http://localhost:5000"
    FRONTEND_URL = "http://localhost:3000"
    
    # Server commands
    BACKEND_CMD = "python app.py"
    FRONTEND_CMD = "npm run dev"
    
    # Test commands
    PYTEST_CMD = "pytest"
    
    # Timeouts (seconds)
    SERVER_STARTUP_TIMEOUT = 30
    SERVER_CHECK_INTERVAL = 1


# ============================================================================
# COLOR OUTPUT
# ============================================================================

class Colors:
    """Terminal colors"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def print_header(text):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN} {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED} {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}  {text}{Colors.END}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}  {text}{Colors.END}")


# ============================================================================
# SERVER MANAGEMENT
# ============================================================================

class ServerManager:
    """Manage backend and frontend servers"""
    
    def __init__(self):
        self.backend_process = None
        self.frontend_process = None
        self.servers_started = False
    
    def check_server(self, url, name):
        """Check if server is responding"""
        try:
            import requests
            response = requests.get(url, timeout=2)
            return response.status_code in [200, 404]  
        except:
            return False
    
    def wait_for_server(self, url, name, timeout=30):
        """Wait for server to start"""
        print_info(f"Waiting for {name} to start...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.check_server(url, name):
                print_success(f"{name} is ready!")
                return True
            time.sleep(Config.SERVER_CHECK_INTERVAL)
        
        print_error(f"{name} failed to start within {timeout}s")
        return False
    
    def start_backend(self):
        """Start backend server"""
        if not Config.BACKEND_DIR.exists():
            print_warning(f"Backend directory not found: {Config.BACKEND_DIR}")
            print_info("Skipping backend startup (assuming it's already running)")
            return True
        
        print_info(f"Starting backend server...")
        print_info(f"Command: {Config.BACKEND_CMD}")
        print_info(f"Directory: {Config.BACKEND_DIR}")
        
        try:
            # Start backend process
            self.backend_process = subprocess.Popen(
                Config.BACKEND_CMD,
                shell=True,
                cwd=Config.BACKEND_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for it to be ready
            if self.wait_for_server(Config.BACKEND_URL, "Backend", Config.SERVER_STARTUP_TIMEOUT):
                return True
            else:
                self.stop_backend()
                return False
                
        except Exception as e:
            print_error(f"Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start frontend server"""
        print_info(f"Starting frontend server...")
        print_info(f"Command: {Config.FRONTEND_CMD}")
        print_info(f"Directory: {Config.FRONTEND_DIR}")
        
        try:
            # Start frontend process
            self.frontend_process = subprocess.Popen(
                Config.FRONTEND_CMD,
                shell=True,
                cwd=Config.FRONTEND_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            # Wait for it to be ready
            if self.wait_for_server(Config.FRONTEND_URL, "Frontend", Config.SERVER_STARTUP_TIMEOUT):
                return True
            else:
                self.stop_frontend()
                return False
                
        except Exception as e:
            print_error(f"Failed to start frontend: {e}")
            return False
    
    def start_servers(self):
        """Start both servers"""
        print_header("STARTING SERVERS")
        
        # Check if servers are already running
        backend_running = self.check_server(Config.BACKEND_URL, "Backend")
        frontend_running = self.check_server(Config.FRONTEND_URL, "Frontend")
        
        if backend_running and frontend_running:
            print_success("Both servers are already running!")
            self.servers_started = False  # We didn't start them
            return True
        
        # Start backend if needed
        if backend_running:
            print_success("Backend is already running")
        else:
            if not self.start_backend():
                return False
        
        # Start frontend if needed
        if frontend_running:
            print_success("Frontend is already running")
        else:
            if not self.start_frontend():
                return False
        
        self.servers_started = True
        print_success("All servers are ready!")
        return True
    
    def stop_backend(self):
        """Stop backend server"""
        if self.backend_process:
            print_info("Stopping backend server...")
            self.backend_process.terminate()
            try:
                self.backend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.backend_process.kill()
            self.backend_process = None
    
    def stop_frontend(self):
        """Stop frontend server"""
        if self.frontend_process:
            print_info("Stopping frontend server...")
            self.frontend_process.terminate()
            try:
                self.frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.frontend_process.kill()
            self.frontend_process = None
    
    def stop_servers(self):
        """Stop both servers"""
        if self.servers_started:
            print_header("STOPPING SERVERS")
            self.stop_backend()
            self.stop_frontend()
            print_success("Servers stopped")


# ============================================================================
# TEST RUNNER
# ============================================================================

class TestRunner:
    """Run pytest tests"""
    
    def __init__(self):
        self.test_result = None
    
    def run_tests(self, args=None):
        """Run pytest with optional arguments"""
        print_header("RUNNING TESTS")
        
        # Build pytest command
        cmd = [Config.PYTEST_CMD]
        
        # Add default arguments
        cmd.extend([
            '-v',                           # Verbose
            '--tb=short',                   # Short traceback
            '--color=yes',                  # Colored output
        ])
        
        # Add user arguments
        if args:
            cmd.extend(args)
        
        print_info(f"Command: {' '.join(cmd)}")
        print()
        
        # Run tests
        try:
            result = subprocess.run(
                cmd,
                cwd=Config.PROJECT_ROOT,
            )
            
            self.test_result = result.returncode
            return result.returncode == 0
            
        except Exception as e:
            print_error(f"Failed to run tests: {e}")
            self.test_result = 1
            return False
    
    def generate_report(self):
        """Generate HTML report"""
        print_header("GENERATING REPORT")
        
        cmd = [
            Config.PYTEST_CMD,
            '--html=test-report.html',
            '--self-contained-html',
            '--tb=short',
        ]
        
        try:
            subprocess.run(cmd, cwd=Config.PROJECT_ROOT)
            print_success("HTML report generated: test-report.html")
            return True
        except Exception as e:
            print_error(f"Failed to generate report: {e}")
            return False


# ============================================================================
# MAIN RUNNER
# ============================================================================

class FrontendTestRunner:
    """Main test runner orchestrator"""
    
    def __init__(self, auto_start_servers=True, auto_stop_servers=True, 
                 generate_report=False, test_args=None):
        self.auto_start_servers = auto_start_servers
        self.auto_stop_servers = auto_stop_servers
        self.generate_report_flag = generate_report
        self.test_args = test_args or []
        
        self.server_manager = ServerManager()
        self.test_runner = TestRunner()
     
    def check_prerequisites(self):
        """Check if prerequisites are met"""
        print_header("CHECKING PREREQUISITES")
        
        # Check Python
        print_info(f"Python version: {sys.version}")
        
        # Check if pytest is installed
        try:
            result = subprocess.run(
                [Config.PYTEST_CMD, '--version'],
                capture_output=True,
                text=True
            )
            print_success(f"Pytest installed: {result.stdout.strip()}")
        except FileNotFoundError:
            print_error("Pytest not found! Run: pip install -r requirements-test.txt")
            return False
        
        # Check if tests directory exists
        if not Config.TESTS_DIR.exists():
            print_error(f"Tests directory not found: {Config.TESTS_DIR}")
            return False
        
        print_success("All prerequisites met!")
        return True
    
    def run(self):
        """Run the complete test suite"""
        
        try:
            # Check prerequisites
            if not self.check_prerequisites():
                return 1
            
            # Start servers if needed
            if self.auto_start_servers:
                if not self.server_manager.start_servers():
                    print_error("Failed to start servers")
                    return 1
            else:
                print_warning("Skipping server startup (manual mode)")
            
            # Run tests
            success = self.test_runner.run_tests(self.test_args)
            
            # Generate report if requested
            if self.generate_report_flag:
                self.test_runner.generate_report()
            
            # Print summary
            self.print_summary(success)
            
            return 0 if success else 1
            
        except KeyboardInterrupt:
            print_warning("\n\nTests interrupted by user")
            return 1
            
        finally:
            # Stop servers if we started them
            if self.auto_stop_servers:
                self.server_manager.stop_servers()
    
    def print_summary(self, success):
        """Print test summary"""
        print_header("TEST SUMMARY")
        
        if success:
            print_success("ALL TESTS PASSED! ")
        else:
            print_error("SOME TESTS FAILED")
        
        print()
        print_info("Next steps:")
        if not success:
            print("  - Review failed tests above")
            print("  - Check test-report.html for details")
            print("  - Run specific test: pytest tests_pytest/test_login.py")
        else:
            print("  - View coverage: pytest --cov=src --cov-report=html")
            print("  - Run with report: python run_frontend_tests.py --report")
        
        print()


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Frontend Test Suite Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_frontend_tests.py                          # Run all tests
  python run_frontend_tests.py --no-servers             # Don't start servers
  python run_frontend_tests.py --report                 # Generate HTML report
  python run_frontend_tests.py -- -m unit               # Run only unit tests
  python run_frontend_tests.py -- -k test_login        # Run tests matching name
  python run_frontend_tests.py -- tests_pytest/test_login.py  # Run specific file
        """
    )
    
    parser.add_argument(
        '--no-servers',
        action='store_true',
        help='Do not start/stop servers automatically'
    )
    
    parser.add_argument(
        '--report',
        action='store_true',
        help='Generate HTML test report'
    )
    
    parser.add_argument(
        '--keep-servers',
        action='store_true',
        help='Keep servers running after tests'
    )
    
    # Capture remaining args for pytest
    args, pytest_args = parser.parse_known_args()
    
    return args, pytest_args


def main():
    """Main entry point"""
    args, pytest_args = parse_args()
    
    # Create runner
    runner = FrontendTestRunner(
        auto_start_servers=not args.no_servers,
        auto_stop_servers=not args.keep_servers,
        generate_report=args.report,
        test_args=pytest_args
    )
    
    # Run tests
    exit_code = runner.run()
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()