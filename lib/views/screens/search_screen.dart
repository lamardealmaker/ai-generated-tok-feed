import 'package:flutter/material.dart';
import '../../constants.dart';

class SearchScreen extends StatelessWidget {
  const SearchScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Container(
          height: 40,
          decoration: BoxDecoration(
            color: AppColors.darkGrey,
            borderRadius: BorderRadius.circular(8),
          ),
          child: const TextField(
            decoration: InputDecoration(
              hintText: 'Search properties',
              prefixIcon: Icon(Icons.search, color: AppColors.grey),
              border: InputBorder.none,
              contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            ),
            style: TextStyle(color: AppColors.white),
          ),
        ),
      ),
      body: const Center(
        child: Text(
          'Search Coming Soon',
          style: TextStyle(color: AppColors.white),
        ),
      ),
    );
  }
}
