import 'package:flutter/material.dart';
import 'package:get/get.dart';
import '../../constants.dart';
import 'home_screen.dart';
import 'search_screen.dart';
import 'messages_screen.dart';
import 'profile_screen.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({super.key});

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int _selectedIndex = 0;
  static const double _iconSize = 28.0;

  final List<Widget> _screens = [
    const HomeScreen(),
    const SearchScreen(),
    const MessagesScreen(),
    ProfileScreen(),
  ];

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_selectedIndex],
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        backgroundColor: AppColors.background,
        selectedItemColor: AppColors.white,
        unselectedItemColor: AppColors.grey,
        showSelectedLabels: true,
        showUnselectedLabels: true,
        selectedLabelStyle: const TextStyle(fontSize: 11),
        unselectedLabelStyle: const TextStyle(fontSize: 11),
        currentIndex: _selectedIndex,
        onTap: _onItemTapped,
        items: const [
          BottomNavigationBarItem(
            icon: Icon(Icons.home_outlined, size: _iconSize),
            activeIcon: Icon(Icons.home, size: _iconSize),
            label: 'For You',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.search_outlined, size: _iconSize),
            activeIcon: Icon(Icons.search, size: _iconSize),
            label: 'Search',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.message_outlined, size: _iconSize),
            activeIcon: Icon(Icons.message, size: _iconSize),
            label: 'Messages',
          ),
          BottomNavigationBarItem(
            icon: Icon(Icons.person_outline, size: _iconSize),
            activeIcon: Icon(Icons.person, size: _iconSize),
            label: 'Profile',
          ),
        ],
      ),
    );
  }
}
