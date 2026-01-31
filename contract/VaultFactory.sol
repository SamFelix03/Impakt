// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title DisasterReliefVault
 * @dev Individual vault for a specific disaster relief fund
 */
contract DisasterReliefVault {
    string public disasterName;
    uint256 public reliefAmount;
    address public factory;
    address public creator;
    uint256 public totalReceived;
    uint256 public totalWithdrawn;
    
    event DonationReceived(address indexed donor, uint256 amount, uint256 timestamp);
    event FundsWithdrawn(address indexed to, uint256 amount, address indexed by, uint256 timestamp);
    
    constructor(string memory _disasterName, uint256 _reliefAmount, address _creator) {
        disasterName = _disasterName;
        reliefAmount = _reliefAmount;
        factory = msg.sender;
        creator = _creator;
    }
    
    // Modifier to restrict access to creator only
    modifier onlyCreator() {
        require(msg.sender == creator, "Only vault creator can perform this action");
        _;
    }
    
    // Receive ETH donations
    receive() external payable {
        totalReceived += msg.value;
        emit DonationReceived(msg.sender, msg.value, block.timestamp);
    }
    
    // Withdraw specific amount to a specific address (only creator)
    function withdraw(address payable _to, uint256 _amount) public onlyCreator {
        require(_to != address(0), "Invalid recipient address");
        require(_amount > 0, "Amount must be greater than 0");
        require(_amount <= address(this).balance, "Insufficient balance in vault");
        
        totalWithdrawn += _amount;
        
        (bool success, ) = _to.call{value: _amount}("");
        require(success, "Transfer failed");
        
        emit FundsWithdrawn(_to, _amount, msg.sender, block.timestamp);
    }
    
    // Withdraw all funds to a specific address (only creator)
    function withdrawAll(address payable _to) public onlyCreator {
        require(_to != address(0), "Invalid recipient address");
        
        uint256 balance = address(this).balance;
        require(balance > 0, "No funds to withdraw");
        
        totalWithdrawn += balance;
        
        (bool success, ) = _to.call{value: balance}("");
        require(success, "Transfer failed");
        
        emit FundsWithdrawn(_to, balance, msg.sender, block.timestamp);
    }
    
    // Get current balance
    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
    
    // Get vault details
    function getVaultInfo() public view returns (
        string memory,
        uint256,
        uint256,
        uint256,
        uint256,
        address
    ) {
        return (
            disasterName, 
            reliefAmount, 
            totalReceived, 
            totalWithdrawn,
            address(this).balance,
            creator
        );
    }
}

/**
 * @title DisasterReliefFactory
 * @dev Factory contract to create and manage disaster relief vaults
 */
contract DisasterReliefFactory {
    // Array to store all vault addresses
    address[] public vaults;
    
    // Mapping from disaster name to vault address
    mapping(string => address) public disasterToVault;
    
    // Mapping from creator to their vaults
    mapping(address => address[]) public creatorVaults;
    
    // Events
    event VaultCreated(
        address indexed vaultAddress,
        string disasterName,
        uint256 reliefAmount,
        address indexed creator,
        uint256 timestamp
    );
    
    /**
     * @dev Create a new disaster relief vault
     * @param _disasterName Name of the disaster
     * @param _reliefAmount Target relief amount needed
     * @return Address of the newly created vault
     */
    function createVault(
        string memory _disasterName,
        uint256 _reliefAmount
    ) public returns (address) {
        // Create new vault with msg.sender as creator
        DisasterReliefVault newVault = new DisasterReliefVault(
            _disasterName,
            _reliefAmount,
            msg.sender
        );
        
        address vaultAddress = address(newVault);
        
        // Store vault address
        vaults.push(vaultAddress);
        disasterToVault[_disasterName] = vaultAddress;
        creatorVaults[msg.sender].push(vaultAddress);
        
        emit VaultCreated(
            vaultAddress,
            _disasterName,
            _reliefAmount,
            msg.sender,
            block.timestamp
        );
        
        return vaultAddress;
    }
    
    /**
     * @dev Get total number of vaults created
     */
    function getVaultCount() public view returns (uint256) {
        return vaults.length;
    }
    
    /**
     * @dev Get vault address by index
     */
    function getVaultByIndex(uint256 index) public view returns (address) {
        require(index < vaults.length, "Index out of bounds");
        return vaults[index];
    }
    
    /**
     * @dev Get vault address by disaster name
     */
    function getVaultByDisaster(string memory _disasterName) 
        public 
        view 
        returns (address) 
    {
        return disasterToVault[_disasterName];
    }
    
    /**
     * @dev Get all vaults created by a specific address
     */
    function getVaultsByCreator(address _creator) 
        public 
        view 
        returns (address[] memory) 
    {
        return creatorVaults[_creator];
    }
    
    /**
     * @dev Get balance of a specific vault
     */
    function getVaultBalance(address _vaultAddress) 
        public 
        view 
        returns (uint256) 
    {
        DisasterReliefVault vault = DisasterReliefVault(payable(_vaultAddress));
        return vault.getBalance();
    }
    
    /**
     * @dev Get all vault balances
     * @return Array of vault addresses and their corresponding balances
     */
    function getAllVaultBalances() 
        public 
        view 
        returns (address[] memory, uint256[] memory) 
    {
        uint256[] memory balances = new uint256[](vaults.length);
        
        for (uint256 i = 0; i < vaults.length; i++) {
            DisasterReliefVault vault = DisasterReliefVault(payable(vaults[i]));
            balances[i] = vault.getBalance();
        }
        
        return (vaults, balances);
    }
    
    /**
     * @dev Get detailed info for all vaults
     * @return addresses Array of vault addresses
     * @return disasterNames Array of disaster names
     * @return reliefAmounts Array of target relief amounts
     * @return currentBalances Array of current vault balances
     */
    function getAllVaultDetails() 
        public 
        view 
        returns (
            address[] memory addresses,
            string[] memory disasterNames,
            uint256[] memory reliefAmounts,
            uint256[] memory currentBalances
        ) 
    {
        addresses = new address[](vaults.length);
        disasterNames = new string[](vaults.length);
        reliefAmounts = new uint256[](vaults.length);
        currentBalances = new uint256[](vaults.length);
        
        for (uint256 i = 0; i < vaults.length; i++) {
            DisasterReliefVault vault = DisasterReliefVault(payable(vaults[i]));
            
            addresses[i] = vaults[i];
            (
                disasterNames[i],
                reliefAmounts[i],
                ,
                ,
                currentBalances[i],
            ) = vault.getVaultInfo();
        }
        
        return (addresses, disasterNames, reliefAmounts, currentBalances);
    }
}