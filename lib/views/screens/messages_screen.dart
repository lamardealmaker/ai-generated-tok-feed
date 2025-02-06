import 'package:flutter/material.dart';
import '../../constants.dart';

class MessagesScreen extends StatelessWidget {
  const MessagesScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: const Text(
          'Messages',
          style: TextStyle(color: AppColors.white),
        ),
      ),
      body: const Center(
        child: Text(
          'Messages Coming Soon',
          style: TextStyle(color: AppColors.white),
        ),
      ),
    );
  }
}
