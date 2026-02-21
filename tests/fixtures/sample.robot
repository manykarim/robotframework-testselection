*** Settings ***
Library    Collections
Library    String

*** Variables ***
${BASE_URL}    https://example.com
${ADMIN_USER}    admin
${ADMIN_PASS}    secret

*** Keywords ***
Open Application
    [Arguments]    ${url}
    Log    Opening application at ${url}

Login As User
    [Arguments]    ${username}    ${password}
    Open Application    ${BASE_URL}
    Input Credentials    ${username}    ${password}
    Submit Login Form

Input Credentials
    [Arguments]    ${username}    ${password}
    Log    Entering username: ${username}
    Log    Entering password: ${password}

Submit Login Form
    Log    Submitting login form

Verify Dashboard
    Log    Verifying dashboard is displayed

Navigate To Settings
    Log    Navigating to settings page

Change Language
    [Arguments]    ${language}
    Navigate To Settings
    Log    Changing language to ${language}

Search For Item
    [Arguments]    ${query}
    Log    Searching for: ${query}

Verify Search Results
    [Arguments]    ${expected_count}
    Log    Verifying ${expected_count} results found

Add Item To Cart
    [Arguments]    ${item_name}
    Search For Item    ${item_name}
    Log    Adding ${item_name} to cart

Verify Cart Total
    [Arguments]    ${expected_total}
    Log    Verifying cart total is ${expected_total}

Complete Checkout
    Log    Completing checkout process

Generate Report
    [Arguments]    ${report_type}
    Log    Generating ${report_type} report

Export Report
    [Arguments]    ${format}
    Log    Exporting report as ${format}

*** Test Cases ***
Login With Valid Credentials
    [Tags]    smoke    authentication
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Verify Dashboard

Login With Invalid Password
    [Tags]    authentication    negative
    Login As User    ${ADMIN_USER}    wrongpassword

Login With Empty Username
    [Tags]    authentication    negative    boundary
    Login As User    ${EMPTY}    ${ADMIN_PASS}

Change UI Language To French
    [Tags]    settings    localization
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Change Language    French

Change UI Language To German
    [Tags]    settings    localization
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Change Language    German

Search Products By Name
    [Tags]    search    smoke
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Search For Item    laptop
    Verify Search Results    5

Search With No Results
    [Tags]    search    boundary
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Search For Item    nonexistent_item_xyz
    Verify Search Results    0

Add Single Item To Cart
    [Tags]    cart    smoke
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Add Item To Cart    Wireless Mouse
    Verify Cart Total    29.99

Add Multiple Items To Cart
    [Tags]    cart    regression
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Add Item To Cart    Wireless Mouse
    Add Item To Cart    USB Keyboard
    Verify Cart Total    59.98

Complete Purchase Flow
    [Tags]    checkout    smoke    e2e
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Add Item To Cart    Laptop Stand
    Complete Checkout

Generate Sales Report
    [Tags]    reporting
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Generate Report    sales

Generate Inventory Report
    [Tags]    reporting
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Generate Report    inventory

Export Report As PDF
    [Tags]    reporting    export
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Generate Report    sales
    Export Report    PDF

Export Report As CSV
    [Tags]    reporting    export
    Login As User    ${ADMIN_USER}    ${ADMIN_PASS}
    Generate Report    inventory
    Export Report    CSV
