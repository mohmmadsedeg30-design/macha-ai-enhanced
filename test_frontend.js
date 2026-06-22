#!/usr/bin/env node

/**
 * Frontend Test Script for Macha AI Chat
 * Tests message sending and receiving
 */

const http = require('http');

function makeRequest(path, method = 'GET', data = null) {
  return new Promise((resolve, reject) => {
    const options = {
      hostname: 'localhost',
      port: 5000,
      path: path,
      method: method,
      headers: {
        'Content-Type': 'application/json'
      }
    };

    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          resolve({
            status: res.statusCode,
            data: JSON.parse(body)
          });
        } catch (e) {
          resolve({
            status: res.statusCode,
            data: body
          });
        }
      });
    });

    req.on('error', reject);
    if (data) req.write(JSON.stringify(data));
    req.end();
  });
}

async function runTests() {
  console.log('\n🧪 Macha AI Frontend Tests\n');
  console.log('=' .repeat(50));

  try {
    // Test 1: Check server is running
    console.log('\n✓ Test 1: Server Connection');
    const homeRes = await makeRequest('/');
    console.log(`  Status: ${homeRes.status} ${homeRes.status === 200 ? '✅' : '❌'}`);

    // Test 2: Chat API - English
    console.log('\n✓ Test 2: Chat API - English Message');
    const chatRes = await makeRequest('/api/chat', 'POST', {
      message: 'Hello, how are you?'
    });
    console.log(`  Status: ${chatRes.status} ${chatRes.status === 200 ? '✅' : '❌'}`);
    console.log(`  Response: ${chatRes.data.response?.substring(0, 50)}...`);
    console.log(`  Source: ${chatRes.data.source}`);

    // Test 3: Chat API - Arabic
    console.log('\n✓ Test 3: Chat API - Arabic Message');
    const arabicRes = await makeRequest('/api/chat', 'POST', {
      message: 'مرحبا، كيف حالك؟'
    });
    console.log(`  Status: ${arabicRes.status} ${arabicRes.status === 200 ? '✅' : '❌'}`);
    console.log(`  Response: ${arabicRes.data.response?.substring(0, 50)}...`);
    console.log(`  Language: ${arabicRes.data.language}`);

    // Test 4: Stats API
    console.log('\n✓ Test 4: Stats API');
    const statsRes = await makeRequest('/api/stats');
    console.log(`  Status: ${statsRes.status} ${statsRes.status === 200 ? '✅' : '❌'}`);
    console.log(`  Device: ${statsRes.data.device}`);
    console.log(`  Model Loaded: ${statsRes.data.model_loaded}`);

    // Test 5: Feedback API
    console.log('\n✓ Test 5: Feedback API');
    const feedbackRes = await makeRequest('/api/feedback', 'POST', {
      input: 'What is AI?',
      wrong_output: 'AI is nothing',
      correct_output: 'AI is artificial intelligence'
    });
    console.log(`  Status: ${feedbackRes.status} ${feedbackRes.status === 200 ? '✅' : '❌'}`);
    console.log(`  Message: ${feedbackRes.data.message}`);

    console.log('\n' + '='.repeat(50));
    console.log('\n✅ All tests completed!\n');

  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
    process.exit(1);
  }
}

runTests();
