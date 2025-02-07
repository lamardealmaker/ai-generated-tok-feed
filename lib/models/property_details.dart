class PropertyDetails {
  final String address;
  final String city;
  final String state;
  final String zipCode;
  final double price;
  final int beds;
  final int baths;
  final int squareFeet;
  final String description;
  final List<String> features;
  final String agentName;
  final String agencyName;

  PropertyDetails({
    required this.address,
    required this.city,
    required this.state,
    required this.zipCode,
    required this.price,
    required this.beds,
    required this.baths,
    required this.squareFeet,
    required this.description,
    required this.features,
    required this.agentName,
    required this.agencyName,
  });

  // Create from Map (for Firestore)
  factory PropertyDetails.fromMap(Map<String, dynamic> map) {
    return PropertyDetails(
      address: map['address'] ?? '',
      city: map['city'] ?? '',
      state: map['state'] ?? '',
      zipCode: map['zipCode'] ?? '',
      price: (map['price'] ?? 0.0).toDouble(),
      beds: (map['beds'] ?? 0).toInt(),
      baths: (map['baths'] ?? 0).toInt(),
      squareFeet: (map['squareFeet'] ?? 0).toInt(),
      description: map['description'] ?? '',
      features: List<String>.from(map['features'] ?? []),
      agentName: map['agentName'] ?? '',
      agencyName: map['agencyName'] ?? '',
    );
  }

  // Convert to Map (for Firestore)
  Map<String, dynamic> toMap() {
    return {
      'address': address,
      'city': city,
      'state': state,
      'zipCode': zipCode,
      'price': price,
      'beds': beds,
      'baths': baths,
      'squareFeet': squareFeet,
      'description': description,
      'features': features,
      'agentName': agentName,
      'agencyName': agencyName,
    };
  }

  // Helper method to format full address
  String get fullAddress => '$address, $city, $state $zipCode';

  // Format price with commas and dollar sign
  String get formattedPrice => '\$${price.toStringAsFixed(0).replaceAllMapped(
    RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'),
    (Match m) => '${m[1]},'
  )}';

  // Copy with method for immutability
  PropertyDetails copyWith({
    String? address,
    String? city,
    String? state,
    String? zipCode,
    double? price,
    int? beds,
    int? baths,
    int? squareFeet,
    String? description,
    List<String>? features,
    String? agentName,
    String? agencyName,
  }) {
    return PropertyDetails(
      address: address ?? this.address,
      city: city ?? this.city,
      state: state ?? this.state,
      zipCode: zipCode ?? this.zipCode,
      price: price ?? this.price,
      beds: beds ?? this.beds,
      baths: baths ?? this.baths,
      squareFeet: squareFeet ?? this.squareFeet,
      description: description ?? this.description,
      features: features ?? this.features,
      agentName: agentName ?? this.agentName,
      agencyName: agencyName ?? this.agencyName,
    );
  }

  Map<String, dynamic> toJson() => {
    'address': address,
    'city': city,
    'state': state,
    'zipCode': zipCode,
    'price': price,
    'beds': beds,
    'baths': baths,
    'squareFeet': squareFeet,
    'description': description,
    'features': features,
    'agentName': agentName,
    'agencyName': agencyName,
  };
}
