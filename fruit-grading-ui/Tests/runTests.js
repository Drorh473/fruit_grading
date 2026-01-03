#!/usr/bin/env node

/**
 * Automated Test Runner
 * Runs comprehensive frontend test suite with reporting
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

// Test configuration
const TEST_SUITES = {
  unit: {
    name: 'Unit Tests',
    command: 'vitest run Tests/context Tests/utils/testUtils.js',
    description: 'Individual component and utility tests',
  },
  components: {
    name: 'Component Tests',
    command: 'vitest run Tests/pages Tests/components',
    description: 'React component rendering and interaction tests',
  },
  api: {
    name: 'API Tests',
    command: 'vitest run Tests/utils/api.test.js',
    description: 'API integration and error handling tests',
  },
  integration: {
    name: 'Integration Tests',
    command: 'vitest run Tests/integration',
    description: 'End-to-end user workflow tests',
  },
};

class TestRunner {
  constructor() {
    this.results = {
      passed: [],
      failed: [],
      skipped: [],
      startTime: Date.now(),
      endTime: null,
    };
  }

  log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
  }

  header(text) {
    const line = '='.repeat(70);
    this.log('\n' + line, 'cyan');
    this.log(text.toUpperCase(), 'bright');
    this.log(line, 'cyan');
  }

  section(text) {
    this.log(`\n${'─'.repeat(70)}`, 'blue');
    this.log(text, 'bright');
    this.log('─'.repeat(70), 'blue');
  }

  runSuite(suiteKey, suite) {
    this.section(`Running ${suite.name}`);
    this.log(suite.description, 'cyan');
    console.log();

    try {
      execSync(suite.command, { stdio: 'inherit' });
      this.results.passed.push(suiteKey);
      this.log(`✓ ${suite.name} PASSED`, 'green');
      return true;
    } catch (error) {
      this.results.failed.push(suiteKey);
      this.log(`✗ ${suite.name} FAILED`, 'red');
      return false;
    }
  }

  runCoverage() {
    this.section('Generating Coverage Report');
    
    try {
      execSync('vitest run --coverage', { stdio: 'inherit' });
      this.log('Coverage report generated', 'green');
      this.log('\nView coverage at: coverage/index.html', 'cyan');
      return true;
    } catch (error) {
      this.log('Coverage generation failed', 'red');
      return false;
    }
  }

  printSummary() {
    this.results.endTime = Date.now();
    const duration = ((this.results.endTime - this.results.startTime) / 1000).toFixed(2);

    this.header('Test Execution Summary');

    // Results
    console.log();
    this.log(`Total Suites: ${Object.keys(TEST_SUITES).length}`, 'bright');
    this.log(`Passed: ${this.results.passed.length}`, 'green');
    this.log(`Failed: ${this.results.failed.length}`, 'red');
    this.log(`Duration: ${duration}s`, 'cyan');

    // Detailed results
    if (this.results.passed.length > 0) {
      console.log();
      this.log('Passed Suites:', 'green');
      this.results.passed.forEach(suite => {
        this.log(`   ${TEST_SUITES[suite].name}`, 'green');
      });
    }

    if (this.results.failed.length > 0) {
      console.log();
      this.log('Failed Suites:', 'red');
      this.results.failed.forEach(suite => {
        this.log(`   ${TEST_SUITES[suite].name}`, 'red');
      });
    }

    console.log();

    // Final status
    if (this.results.failed.length === 0) {
      this.log(' ALL TESTS PASSED!', 'green');
      this.log('The frontend is ready for deployment.', 'cyan');
      return 0;
    } else {
      this.log('  SOME TESTS FAILED', 'red');
      this.log('Please fix failing tests before deployment.', 'yellow');
      return 1;
    }
  }

  async run(options = {}) {
    this.header('Fruit Grading System - Frontend Test Suite');

    // Run individual suites
    for (const [key, suite] of Object.entries(TEST_SUITES)) {
      if (options.suite && options.suite !== key) continue;
      this.runSuite(key, suite);
    }

    // Run coverage if requested
    if (options.coverage) {
      this.runCoverage();
    }

    // Print summary
    const exitCode = this.printSummary();

    // Save results to file
    if (options.saveResults) {
      this.saveResults();
    }

    return exitCode;
  }

  saveResults() {
    const resultsPath = path.join(__dirname, 'test-results.json');
    const results = {
      ...this.results,
      timestamp: new Date().toISOString(),
      duration: (this.results.endTime - this.results.startTime) / 1000,
    };

    fs.writeFileSync(resultsPath, JSON.stringify(results, null, 2));
    this.log(`\nResults saved to: ${resultsPath}`, 'cyan');
  }
}

// Parse command line arguments
const args = process.argv.slice(2);
const options = {
  coverage: args.includes('--coverage'),
  saveResults: args.includes('--save-results'),
  suite: args.find(arg => arg.startsWith('--suite='))?.split('=')[1],
};

// Run tests
const runner = new TestRunner();
runner.run(options).then(exitCode => {
  process.exit(exitCode);
});
